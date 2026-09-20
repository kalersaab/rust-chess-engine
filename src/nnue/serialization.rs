use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;
use ndarray::{Array1, Array2};
use super::architecture::*;

pub struct NNUESerializer;

impl NNUESerializer {
    pub fn save<P: AsRef<Path>>(weights: &NNUEWeights, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        Self::write_array2(&mut writer, &weights.input_weights)?;
        Self::write_array1(&mut writer, &weights.input_bias)?;
        Self::write_array2(&mut writer, &weights.hidden_weights)?;
        Self::write_array1(&mut writer, &weights.hidden_bias)?;
        Self::write_array2(&mut writer, &weights.output_weights)?;
        writer.write_all(&weights.output_bias.to_le_bytes())
            .map_err(|e| format!("Failed to write output bias: {}", e))?;

        writer.flush().map_err(|e| format!("Failed to flush: {}", e))?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<NNUEWeights, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        let input_weights = Self::read_array2(&mut reader, HIDDEN_SIZE, INPUT_SIZE)?;
        let input_bias = Self::read_array1(&mut reader, HIDDEN_SIZE)?;
        let hidden_weights = Self::read_array2(&mut reader, OUTPUT_SIZE, HIDDEN_SIZE)?;
        let hidden_bias = Self::read_array1(&mut reader, HIDDEN_SIZE)?;
        let output_weights = Self::read_array2(&mut reader, OUTPUT_SIZE, HIDDEN_SIZE)?;

        let mut output_bias_bytes = [0u8; 4];
        reader.read_exact(&mut output_bias_bytes)
            .map_err(|e| format!("Failed to read output bias: {}", e))?;
        let output_bias = f32::from_le_bytes(output_bias_bytes);

        Ok(NNUEWeights {
            input_weights,
            input_bias,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        })
    }

    fn write_array2<W: Write>(writer: &mut W, array: &Array2<f32>) -> Result<(), String> {
        for val in array.iter() {
            writer.write_all(&val.to_le_bytes())
                .map_err(|e| format!("Failed to write array2 element: {}", e))?;
        }
        Ok(())
    }

    fn write_array1<W: Write>(writer: &mut W, array: &Array1<f32>) -> Result<(), String> {
        for val in array.iter() {
            writer.write_all(&val.to_le_bytes())
                .map_err(|e| format!("Failed to write array1 element: {}", e))?;
        }
        Ok(())
    }

    fn read_array2<R: Read>(
        reader: &mut R,
        rows: usize,
        cols: usize,
    ) -> Result<Array2<f32>, String> {
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows * cols {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)
                .map_err(|e| format!("Failed to read array2 element: {}", e))?;
            data[i] = f32::from_le_bytes(bytes);
        }
        Ok(Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| format!("Failed to create array2: {}", e))?)
    }

    fn read_array1<R: Read>(reader: &mut R, size: usize) -> Result<Array1<f32>, String> {
        let mut data = vec![0.0; size];
        for i in 0..size {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)
                .map_err(|e| format!("Failed to read array1 element: {}", e))?;
            data[i] = f32::from_le_bytes(bytes);
        }
        Ok(Array1::from_vec(data))
    }
}
