use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const SF_LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

#[derive(Debug, Clone)]
pub struct SFNNUEInfo {
    pub file_size: usize,
    pub version: u32,
    pub architecture_hash: u32,
    pub description: String,
    pub feature_transformer_hash: u32,
    pub ft_biases_count: usize,
    pub ft_sample_biases: Vec<i16>,
    pub threat_weights_bytes: usize,
    pub threat_psqt_count: usize,
    pub pawn_pair_weights_bytes: usize,
    pub pawn_pair_psqt_count: usize,
    pub half_ka_weights_count: usize,
    pub half_ka_psqt_count: usize,
    pub network_architecture_hash: u32,
    pub layer_stacks_count: usize,
    pub fc0_dim: (usize, usize),
    pub fc1_dim: (usize, usize),
    pub fc2_dim: (usize, usize),
    pub bytes_consumed: usize,
    pub is_valid: bool,
}

pub struct SFNNUEProbe;

impl SFNNUEProbe {
    /// Probe and validate a Stockfish NNUE binary file.
    pub fn probe_file<P: AsRef<Path>>(path: P) -> Result<SFNNUEInfo, String> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to open NNUE file '{}': {}", path.as_ref().display(), e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read NNUE file: {}", e))?;

        Self::probe_slice(&data)
    }

    /// Inspect and validate raw binary bytes of a Stockfish NNUE file.
    pub fn probe_slice(data: &[u8]) -> Result<SFNNUEInfo, String> {
        let file_size = data.len();
        if file_size < 12 {
            return Err(format!("File too short ({} bytes)", file_size));
        }

        let mut offset = 0;

        let version = Self::read_u32(data, &mut offset)?;
        let architecture_hash = Self::read_u32(data, &mut offset)?;
        let desc_len = Self::read_u32(data, &mut offset)? as usize;

        if offset + desc_len > file_size {
            return Err("Description length exceeds file boundaries".to_string());
        }

        let description = String::from_utf8_lossy(&data[offset..offset + desc_len]).to_string();
        offset += desc_len;

        let feature_transformer_hash = Self::read_u32(data, &mut offset)?;
        let expected_net_arch_hash = architecture_hash ^ feature_transformer_hash;

        // 1. Feature Transformer Biases (1024 int16 in LEB128)
        let (ft_biases_count, sample_biases) =
            Self::read_leb128_block(data, &mut offset, 1024, 8)?;

        // 2. Full Threats weights (59,808 * 1024 int8 raw)
        let threat_weights_bytes = 59808 * 1024;
        if offset + threat_weights_bytes > file_size {
            return Err("Threat weights exceed file boundaries".to_string());
        }
        offset += threat_weights_bytes;

        // 3. Full Threats PSQT (59,808 * 8 int32 in LEB128)
        let (threat_psqt_count, _) =
            Self::read_leb128_block(data, &mut offset, 59808 * 8, 0)?;

        // 4. Pawn Pair weights (4,560 * 1024 int8 raw)
        let pawn_pair_weights_bytes = 4560 * 1024;
        if offset + pawn_pair_weights_bytes > file_size {
            return Err("Pawn pair weights exceed file boundaries".to_string());
        }
        offset += pawn_pair_weights_bytes;

        // 5. Pawn Pair PSQT (4,560 * 8 int32 in LEB128)
        let (pawn_pair_psqt_count, _) =
            Self::read_leb128_block(data, &mut offset, 4560 * 8, 0)?;

        // 6. HalfKAv2 weights (22,528 * 1024 int16 in LEB128)
        let (half_ka_weights_count, _) =
            Self::read_leb128_block(data, &mut offset, 22528 * 1024, 0)?;

        // 7. HalfKAv2 PSQT (22,528 * 8 int32 in LEB128)
        let (half_ka_psqt_count, _) =
            Self::read_leb128_block(data, &mut offset, 22528 * 8, 0)?;

        // 8 Layer Stacks of NetworkArchitecture
        let layer_stacks_count = 8;
        let fc0_dim = (1024, 32);
        let fc1_dim = (64, 32);
        let fc2_dim = (128, 1);

        for stack_idx in 0..layer_stacks_count {
            let stack_hash = Self::read_u32(data, &mut offset)?;
            if stack_hash != expected_net_arch_hash {
                return Err(format!(
                    "Layer stack {} hash mismatch: expected 0x{:08x}, got 0x{:08x}",
                    stack_idx, expected_net_arch_hash, stack_hash
                ));
            }

            // fc_0: biases (32 * 4 bytes) + weights (32 * 1024 bytes)
            let fc0_bytes = 32 * 4 + 32 * 1024;
            if offset + fc0_bytes > file_size {
                return Err(format!("Stack {} fc_0 exceeds file size", stack_idx));
            }
            offset += fc0_bytes;

            // fc_1: biases (32 * 4 bytes) + weights (32 * 64 bytes)
            let fc1_bytes = 32 * 4 + 32 * 64;
            if offset + fc1_bytes > file_size {
                return Err(format!("Stack {} fc_1 exceeds file size", stack_idx));
            }
            offset += fc1_bytes;

            // fc_2: biases (1 * 4 bytes) + weights (1 * 128 bytes)
            let fc2_bytes = 1 * 4 + 1 * 128;
            if offset + fc2_bytes > file_size {
                return Err(format!("Stack {} fc_2 exceeds file size", stack_idx));
            }
            offset += fc2_bytes;
        }

        let is_valid = offset == file_size;

        Ok(SFNNUEInfo {
            file_size,
            version,
            architecture_hash,
            description,
            feature_transformer_hash,
            ft_biases_count,
            ft_sample_biases: sample_biases,
            threat_weights_bytes,
            threat_psqt_count,
            pawn_pair_weights_bytes,
            pawn_pair_psqt_count,
            half_ka_weights_count,
            half_ka_psqt_count,
            network_architecture_hash: expected_net_arch_hash,
            layer_stacks_count,
            fc0_dim,
            fc1_dim,
            fc2_dim,
            bytes_consumed: offset,
            is_valid,
        })
    }

    fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, String> {
        if *offset + 4 > data.len() {
            return Err("Unexpected EOF while reading u32".to_string());
        }
        let bytes: [u8; 4] = data[*offset..*offset + 4]
            .try_into()
            .map_err(|_| "Failed slice conversion".to_string())?;
        *offset += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_leb128_block(
        data: &[u8],
        offset: &mut usize,
        expected_count: usize,
        sample_count: usize,
    ) -> Result<(usize, Vec<i16>), String> {
        let magic_len = SF_LEB128_MAGIC.len();
        if *offset + magic_len > data.len() {
            return Err("Unexpected EOF reading LEB128 magic".to_string());
        }
        if &data[*offset..*offset + magic_len] != SF_LEB128_MAGIC {
            return Err(format!(
                "Invalid LEB128 magic string at byte offset {}",
                *offset
            ));
        }
        *offset += magic_len;

        let byte_count = Self::read_u32(data, offset)? as usize;
        let payload_end = *offset + byte_count;
        if payload_end > data.len() {
            return Err(format!(
                "LEB128 compressed block length {} exceeds file boundaries",
                byte_count
            ));
        }

        let mut samples = Vec::new();
        if sample_count > 0 {
            let mut curr = *offset;
            for _ in 0..sample_count.min(expected_count) {
                if curr >= payload_end {
                    break;
                }
                let (val, next_curr) = Self::decode_signed_leb128(&data, curr, payload_end)?;
                samples.push(val as i16);
                curr = next_curr;
            }
        }

        *offset = payload_end;
        Ok((expected_count, samples))
    }

    fn decode_signed_leb128(data: &[u8], mut curr: usize, end: usize) -> Result<(i32, usize), String> {
        let mut result: u32 = 0;
        let mut shift = 0;

        loop {
            if curr >= end {
                return Err("LEB128 stream terminated prematurely".to_string());
            }
            let byte = data[curr];
            curr += 1;

            result |= ((byte & 0x7f) as u32) << (shift % 32);
            shift += 7;

            if (byte & 0x80) == 0 {
                let signed = if shift < 32 && (byte & 0x40) != 0 {
                    (result | (!0u32 << shift)) as i32
                } else {
                    result as i32
                };
                return Ok((signed, curr));
            }
        }
    }
}
