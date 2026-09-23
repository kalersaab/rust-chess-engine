use std::fs::File;
use std::io::Read;
use std::path::Path;
use ndarray::{Array1, Array2};
use super::architecture::*;
use super::sf_probe::{SFNNUEProbe, SFNNUEInfo, SF_LEB128_MAGIC};

pub struct SFNNUELoader;

impl SFNNUELoader {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<NNUEWeights, String> {
        let info = SFNNUEProbe::probe_file(&path)?;
        
        if !info.is_valid {
            return Err(format!(
                "Invalid Stockfish NNUE file: consumed {} bytes but file is {} bytes",
                info.bytes_consumed, info.file_size
            ));
        }

        eprintln!("Loading Stockfish NNUE: {}", info.description);
        eprintln!("  Version: 0x{:08x}", info.version);
        eprintln!("  Architecture: 0x{:08x}", info.architecture_hash);
        eprintln!("  Feature Transformer: {} biases", info.ft_biases_count);
        eprintln!("  Layer Stacks: {}", info.layer_stacks_count);
        eprintln!("  FC0: {:?}, FC1: {:?}, FC2: {:?}", info.fc0_dim, info.fc1_dim, info.fc2_dim);

        let mut file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to reopen file: {}", e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        Self::parse_weights(&data, &info)
    }

    fn parse_weights(data: &[u8], _info: &SFNNUEInfo) -> Result<NNUEWeights, String> {
        let mut offset = 0;
        offset += 4; // version
        offset += 4; // arch hash
        let desc_len = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4 + desc_len;
        offset += 4; // ft hash

        // Read feature transformer biases (1024 i16 in LEB128)
        let ft_biases = Self::read_leb128_i16_array(&data, &mut offset, 1024)?;

        // Skip threat weights (59,808 * 1024 bytes)
        offset += 59808 * 1024;

        // Skip threat PSQT (LEB128 block)
        offset = Self::skip_leb128_block(&data, offset)?;

        // Skip pawn pair weights (4,560 * 1024 bytes)
        offset += 4560 * 1024;

        // Skip pawn pair PSQT (LEB128 block)
        offset = Self::skip_leb128_block(&data, offset)?;

        // Read HalfKAv2 weights (22,528 * 1024 i16 in LEB128)
        let half_ka_weights = Self::read_leb128_i16_array(&data, &mut offset, 22528 * 1024)?;

        // Skip HalfKAv2 PSQT (LEB128 block)
        offset = Self::skip_leb128_block(&data, offset)?;

        // Read first layer stack (stack 0)
        offset += 4; // stack hash

        // FC0: 32 biases (f32) + 32x1024 weights (i8)
        let fc0_biases = Self::read_f32_array(&data, &mut offset, 32)?;
        let fc0_weights = Self::read_i8_array(&data, &mut offset, 32 * 1024)?;

        // FC1: 32 biases (f32) + 32x64 weights (i8)
        let fc1_biases = Self::read_f32_array(&data, &mut offset, 32)?;
        let fc1_weights = Self::read_i8_array(&data, &mut offset, 32 * 64)?;

        // FC2: 1 bias (f32) + 1x128 weights (i8)
        let fc2_biases = Self::read_f32_array(&data, &mut offset, 1)?;
        let fc2_weights = Self::read_i8_array(&data, &mut offset, 1 * 128)?;

        // Convert to our architecture (768->32768->1)
        Self::convert_to_our_format(
            &ft_biases,
            &half_ka_weights,
            &fc0_biases,
            &fc0_weights,
            &fc1_biases,
            &fc1_weights,
            &fc2_biases,
            &fc2_weights,
        )
    }

    fn convert_to_our_format(
        ft_biases: &[i16],
        half_ka_weights: &[i16],
        fc0_biases: &[f32],
        _fc0_weights: &[i8],
        _fc1_biases: &[f32],
        _fc1_weights: &[i8],
        fc2_biases: &[f32],
        _fc2_weights: &[i8],
    ) -> Result<NNUEWeights, String> {
        // Our architecture: 768 -> 32768 -> 1
        // SF architecture: complex HalfKAv2 -> 1024 -> 32 -> 32 -> 1

        // Approximate conversion: Use first 768 features from HalfKA
        // and map SF's layer stack to our hidden layer

        // Input weights: Take first 768 features from half_ka_weights
        // HalfKA has 22528 features, we need 768
        let mut input_weights = Array2::zeros((HIDDEN_SIZE, INPUT_SIZE));
        for h in 0..HIDDEN_SIZE.min(1024) {
            for i in 0..INPUT_SIZE {
                if i < 768 && h * 768 + i < half_ka_weights.len() {
                    input_weights[[h, i]] = half_ka_weights[h * 768 + i] as f32 / 127.0;
                }
            }
        }

        // Input biases: Use feature transformer biases
        let mut input_bias = Array1::zeros(HIDDEN_SIZE);
        for i in 0..HIDDEN_SIZE.min(1024) {
            if i < ft_biases.len() {
                input_bias[i] = ft_biases[i] as f32 / 127.0;
            }
        }

        // Output weights: Combine FC layers
        // FC0: 1024 -> 32, FC1: 64 -> 32, FC2: 128 -> 1
        // We approximate by using FC0 and averaging
        let mut output_weights = Array2::zeros((OUTPUT_SIZE, HIDDEN_SIZE));
        for h in 0..HIDDEN_SIZE.min(1024) {
            let fc0_idx = h % 32;
            if fc0_idx < fc0_biases.len() {
                output_weights[[0, h]] = fc0_biases[fc0_idx] * 0.01;
            }
        }

        // Output bias
        let output_bias = if !fc2_biases.is_empty() {
            fc2_biases[0]
        } else {
            0.0
        };

        Ok(NNUEWeights {
            input_weights,
            input_bias,
            hidden_weights: Array2::zeros((OUTPUT_SIZE, HIDDEN_SIZE)),
            hidden_bias: Array1::zeros(HIDDEN_SIZE),
            output_weights,
            output_bias,
        })
    }

    fn read_leb128_i16_array(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i16>, String> {
        // Check for LEB128 magic
        let magic_len = SF_LEB128_MAGIC.len();
        if *offset + magic_len > data.len() {
            return Err("Unexpected EOF reading LEB128 magic".to_string());
        }
        if &data[*offset..*offset + magic_len] != SF_LEB128_MAGIC {
            return Err("Invalid LEB128 magic".to_string());
        }
        *offset += magic_len;

        // Read byte count
        let byte_count = u32::from_le_bytes(
            data[*offset..*offset+4].try_into().unwrap()
        ) as usize;
        *offset += 4;

        let payload_end = *offset + byte_count;
        let mut result = Vec::with_capacity(count);

        while result.len() < count && *offset < payload_end {
            let (val, next_offset) = Self::decode_signed_leb128(data, *offset, payload_end)?;
            result.push(val as i16);
            *offset = next_offset;
        }

        *offset = payload_end;
        Ok(result)
    }

    fn skip_leb128_block(data: &[u8], mut offset: usize) -> Result<usize, String> {
        let magic_len = SF_LEB128_MAGIC.len();
        if offset + magic_len > data.len() {
            return Err("EOF in skip_leb128".to_string());
        }
        offset += magic_len;

        let byte_count = u32::from_le_bytes(
            data[offset..offset+4].try_into().unwrap()
        ) as usize;
        offset += 4 + byte_count;

        Ok(offset)
    }

    fn read_f32_array(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<f32>, String> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            if *offset + 4 > data.len() {
                return Err("EOF reading f32 array".to_string());
            }
            let bytes: [u8; 4] = data[*offset..*offset+4].try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
            *offset += 4;
        }
        Ok(result)
    }

    fn read_i8_array(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i8>, String> {
        if *offset + count > data.len() {
            return Err("EOF reading i8 array".to_string());
        }
        let result: Vec<i8> = data[*offset..*offset+count]
            .iter()
            .map(|&b| b as i8)
            .collect();
        *offset += count;
        Ok(result)
    }

    fn decode_signed_leb128(data: &[u8], mut curr: usize, end: usize) -> Result<(i32, usize), String> {
        let mut result: u32 = 0;
        let mut shift = 0;

        loop {
            if curr >= end {
                return Err("LEB128 stream terminated".to_string());
            }
            let byte = data[curr];
            curr += 1;

            result |= ((byte & 0x7f) as u32) << shift;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sf_loader_exists() {
        // Test that the loader module exists
        let _loader = SFNNUELoader;
    }
}
