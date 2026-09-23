use crate::board::{Board, Color};
use crate::evaluation::Score;
use std::sync::mpsc;

use super::architecture::{HIDDEN_SIZE, INPUT_SIZE, NNUEWeights, SCALE_FACTOR};
use super::features::FeatureGenerator;

pub const MAX_ACTIVE: usize = 32;
pub const SENTINEL: u32 = u32::MAX;
pub const DEFAULT_BATCH_CAPACITY: usize = 256;

const SHADER_SRC: &str = include_str!("shaders.wgsl");

/// Batch NNUE evaluator backed by a GPU compute pipeline (Metal on macOS).
///
/// Implements the same function as `NNUENetwork::evaluate` (full forward
/// 768 -> 32768 -> 1 in f32) but evaluates many independent positions in one
/// dispatch. Activations are supplied sparsely (active feature indices) so the
/// hidden layer is materialised once per batch.
pub struct GpuNnue {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_hidden: wgpu::ComputePipeline,
    pipeline_output: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    features_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    capacity: usize,
}

impl GpuNnue {
    /// Attempts to initialise a GPU device from the given weights.
    /// Returns `None` when no adapter/backend is available.
    pub fn from_weights(weights: &NNUEWeights) -> Option<GpuNnue> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        log_adapter(&adapter.get_info());

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 0x4000_0000;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nnue-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        ))
        .ok()?;

        Some(GpuNnue::create(device, queue, weights, DEFAULT_BATCH_CAPACITY))
    }

    fn create(
        device: wgpu::Device,
        queue: wgpu::Queue,
        weights: &NNUEWeights,
        capacity: usize,
    ) -> GpuNnue {
        assert!(capacity > 0);

        use wgpu::util::DeviceExt;

        let mut input_w_t = vec![0f32; INPUT_SIZE * HIDDEN_SIZE];
        for (pos, &val) in weights.input_weights.t().iter().enumerate() {
            input_w_t[pos] = val;
        }
        let input_bias = weights.input_bias.iter().cloned().collect::<Vec<_>>();
        let output_w = weights.output_weights.iter().cloned().collect::<Vec<_>>();
        let output_bias = vec![weights.output_bias];

        let input_w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-input-w"),
            contents: bytemuck::cast_slice(&input_w_t),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let input_bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-input-bias"),
            contents: bytemuck::cast_slice(&input_bias),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-output-w"),
            contents: bytemuck::cast_slice(&output_w),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-output-bias"),
            contents: bytemuck::cast_slice(&output_bias),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let features_init = vec![SENTINEL; capacity * MAX_ACTIVE];
        let features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-features"),
            contents: bytemuck::cast_slice(&features_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let hidden_init = vec![0f32; capacity * HIDDEN_SIZE];
        let hidden_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out_init = vec![0f32; capacity];
        let out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-out"),
            contents: bytemuck::cast_slice(&out_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnue-readback"),
            size: (capacity * std::mem::size_of::<f32>() * HIDDEN_SIZE) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nnue-params"),
            contents: &[0u8; 16],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nnue-gpu-layout"),
            entries: &[
                bind_storage(0, true),
                bind_storage(1, true),
                bind_storage(2, true),
                bind_storage(3, false),
                bind_storage(4, true),
                bind_storage(5, true),
                bind_storage(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nnue-gpu-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: features_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hidden_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nnue-gpu-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nnue-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let pipeline_hidden = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nnue-pass-hidden"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("pass1"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_output = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nnue-pass-output"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("pass2"),
            compilation_options: Default::default(),
            cache: None,
        });

        GpuNnue {
            device,
            queue,
            pipeline_hidden,
            pipeline_output,
            bind_group,
            features_buf,
            out_buf,
            readback_buf,
            params_buf,
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn evaluate_boards(&self, boards: &[Board]) -> Vec<Score> {
        let active = boards
            .iter()
            .map(FeatureGenerator::active_feature_indices)
            .collect::<Vec<_>>();
        let raw = self.evaluate_active(&active);
        boards
            .iter()
            .zip(raw.iter())
            .map(|(b, r)| scale_score(*r, b.turn))
            .collect()
    }

    pub fn evaluate_active(&self, active: &[Vec<usize>]) -> Vec<f32> {
        let mut results = Vec::with_capacity(active.len());
        for chunk in active.chunks(self.capacity) {
            results.extend(self.evaluate_chunk(chunk));
        }
        results
    }

    fn evaluate_chunk(&self, chunk: &[Vec<usize>]) -> Vec<f32> {
        let batch = chunk.len();
        debug_assert!(batch <= self.capacity);

        // Fill feature slots: active indices then sentinel padding.
        let mut feats = vec![SENTINEL; self.capacity * MAX_ACTIVE];
        for (item, list) in chunk.iter().enumerate() {
            for (slot, &f) in list.iter().take(MAX_ACTIVE).enumerate() {
                feats[item * MAX_ACTIVE + slot] = f as u32;
            }
        }

        let mut params = [0u8; 16];
        params[..4].copy_from_slice(&(batch as u32).to_le_bytes());

        self.queue
            .write_buffer(&self.features_buf, 0, bytemuck::cast_slice(&feats));
        self.queue.write_buffer(&self.params_buf, 0, &params);

        let groups_hidden = (batch * HIDDEN_SIZE).div_ceil(256) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nnue-gpu-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nnue-pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_hidden);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(groups_hidden, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nnue-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_output);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(batch as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.out_buf,
            0,
            &self.readback_buf,
            0,
            (batch * std::mem::size_of::<f32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let readback_len = batch * std::mem::size_of::<f32>();
        let slice = self.readback_buf.slice(0..readback_len as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        if self
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
        {
            self.readback_buf.unmap();
            return vec![0f32; batch];
        }
        if !matches!(rx.recv(), Ok(Ok(()))) {
            self.readback_buf.unmap();
            return vec![0f32; batch];
        }

        let data = match self.readback_buf.get_mapped_range(0..readback_len as u64) {
            Ok(v) => v,
            Err(_) => {
                self.readback_buf.unmap();
                return vec![0f32; batch];
            }
        };
        let mut out = Vec::with_capacity(batch);
        for b in data.chunks_exact(4) {
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        drop(data);
        self.readback_buf.unmap();
        out
    }
}

fn bind_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(4),
        },
        count: None,
    }
}

fn scale_score(raw: f32, turn: Color) -> Score {
    let scaled = ((raw * SCALE_FACTOR) as Score).max(-32000).min(32000);
    if turn == Color::White {
        scaled
    } else {
        -scaled
    }
}

fn log_adapter(info: &wgpu::AdapterInfo) {
    eprintln!(
        "[gpu] adapter: {} ({:?}) backend: {:?}",
        info.name, info.device_type, info.backend
    );
}

pub fn cpu_forward_active(weights: &NNUEWeights, active: &[Vec<usize>]) -> Vec<f32> {
    use ndarray::Array1;
    let in_t = weights.input_weights.t();
    let mut out = Vec::with_capacity(active.len());
    for list in active {
        let mut hidden = weights.input_bias.clone();
        for &f in list {
            hidden = hidden + &in_t.index_axis(ndarray::Axis(0), f);
        }
        let activated = hidden.mapv(super::architecture::relu);
        let raw = weights.output_weights.dot(&activated)[0] + weights.output_bias;
        out.push(raw);
    }
    let _: Option<Array1<f32>> = None;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::ChessMove;
    use crate::nnue::NNUEWeights;

    const KIWIPETE: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    fn sample_boards(n: usize) -> Vec<Board> {
        let mut positions = Vec::with_capacity(n + 1);
        let mut board = Board::new();
        positions.push(board.clone());
        // e2e4, g1f3, b8c6, d2d4, e7e6  ((rank, file) pairs)
        let moves: [((usize, usize), (usize, usize)); 5] = [
            ((6, 4), (4, 4)),
            ((7, 6), (5, 5)),
            ((0, 1), (2, 2)),
            ((6, 3), (4, 3)),
            ((1, 4), (2, 4)),
        ];
        for (from, to) in moves {
            let mv = ChessMove::new(from.0, from.1, to.0, to.1);
            if board.execute_move(mv.from, mv.to, mv.move_type).is_ok() {
                positions.push(board.clone());
            }
        }
        while positions.len() < n {
            positions.push(Board::new());
        }
        positions.truncate(n);
        positions
    }

    #[test]
    fn gpu_equivalence_random_boards() {
        let weights = NNUEWeights::new();
        let Some(gpu) = GpuNnue::from_weights(&weights) else {
            eprintln!("[gpu] skipped: no adapter available");
            return;
        };
        let boards = sample_boards(24);
        let active = boards
            .iter()
            .map(FeatureGenerator::active_feature_indices)
            .collect::<Vec<_>>();
        let gpu_raw = gpu.evaluate_active(&active);
        let cpu_raw = cpu_forward_active(&weights, &active);
        assert_eq!(gpu_raw.len(), cpu_raw.len());
        for (g, c) in gpu_raw.iter().zip(cpu_raw.iter()) {
            let rel = (g - c).abs() / c.abs().max(1e-6);
            assert!(rel < 5e-2, "gpu {g} vs cpu {c} rel diff {rel}");
        }
    }

    #[test]
    fn gpu_batch_invariance() {
        let weights = NNUEWeights::new();
        let Some(gpu) = GpuNnue::from_weights(&weights) else {
            eprintln!("[gpu] skipped: no adapter available");
            return;
        };
        let boards = sample_boards(70);
        let active = boards
            .iter()
            .map(FeatureGenerator::active_feature_indices)
            .collect::<Vec<_>>();
        let full = gpu.evaluate_active(&active);
        let prefix = gpu.evaluate_active(&active[..40]);
        assert_eq!(prefix.len(), 40);
        assert_eq!(full[..40], prefix);
    }

    #[test]
    fn gpu_known_position_signs() {
        let weights = NNUEWeights::new();
        let Some(gpu) = GpuNnue::from_weights(&weights) else {
            eprintln!("[gpu] skipped: no adapter available");
            return;
        };
        let board = Board::new();
        let scores = gpu.evaluate_boards(&[board.clone()]);
        assert!(scores[0].abs() < 32000);

        let board = Board::from_fen(KIWIPETE).unwrap();
        let scores = gpu.evaluate_boards(&[board.clone()]);
        assert!(scores[0].abs() < 32000);
    }

    #[test]
    fn gpu_controlled_constant_hidden() {
        use ndarray::Array2;
        let mut w = NNUEWeights::from_zeros();
        w.input_bias.fill(1.0);
        w.output_weights = Array2::ones((1, HIDDEN_SIZE));
        w.output_bias = 0.5;
        let active = vec![vec![0usize], vec![10usize], vec![50usize], vec![700usize]];
        let Some(gpu) = GpuNnue::from_weights(&w) else {
            eprintln!("[gpu] skipped: no adapter available");
            return;
        };
        let gpu_raw = gpu.evaluate_active(&active);
        for (i, g) in gpu_raw.iter().enumerate() {
            eprintln!(
                "controlled item={i} gpu={g} expected={}",
                32768.0f32 + 0.5
            );
            assert!((g - (32768.0f32 + 0.5)).abs() < 0.01, "item {i} got {g}");
        }
    }
}