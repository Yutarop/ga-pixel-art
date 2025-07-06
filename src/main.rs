use image::{ImageBuffer, Rgb, RgbImage};
use rand::prelude::*;
use std::f64;
use std::fs::File;
use std::io::BufWriter;

const IMG_SIZE: usize = 100;
const POPULATION_SIZE: usize = 6;
const ITERATION: usize = 50;
const MUTATION_RATE: f64 = 0.05;
const CROSSOVER_RATE: f64 = 0.8;
const GENE_LENGTH: usize = 8;
const RGB_CHANNELS: usize = 3;
const TOURNAMENT_SIZE: usize = 3;
const ELITE_SIZE: usize = 2;

#[derive(Clone, Debug)]
struct Chromosome {
    pos: (usize, usize),
    gene: Vec<Vec<bool>>,
}

impl Chromosome {
    fn new(pos: (usize, usize)) -> Self {
        let mut rng = thread_rng();
        let gene = (0..RGB_CHANNELS)
            .map(|_| {
                (0..GENE_LENGTH)
                    .map(|_| rng.gen_bool(0.5))
                    .collect()
            })
            .collect();

        Chromosome { pos, gene }
    }

    fn mutate(&mut self) {
        let mut rng = thread_rng();
        
        for channel in &mut self.gene {
            for bit in channel {
                if rng.gen::<f64>() < MUTATION_RATE {
                    *bit = !*bit;
                }
            }
        }
        
        if rng.gen::<f64>() < 0.1 {
            let channel_idx = rng.gen_range(0..RGB_CHANNELS);
            let bit_idx = rng.gen_range(0..GENE_LENGTH);
            self.gene[channel_idx][bit_idx] = !self.gene[channel_idx][bit_idx];
        }
    }

    #[allow(dead_code)]
    fn crossover(&self, other: &Chromosome) -> (Chromosome, Chromosome) {
        let mut rng = thread_rng();

        if rng.gen::<f64>() > CROSSOVER_RATE {
            return (self.clone(), other.clone());
        }

        let mut child1 = self.clone();
        let mut child2 = other.clone();

        for i in 0..RGB_CHANNELS {
            let crossover_point = rng.gen_range(1..GENE_LENGTH);
            for j in crossover_point..GENE_LENGTH {
                child1.gene[i][j] = other.gene[i][j];
                child2.gene[i][j] = self.gene[i][j];
            }
        }

        (child1, child2)
    }

    fn uniform_crossover(&self, other: &Chromosome) -> (Chromosome, Chromosome) {
        let mut rng = thread_rng();

        if rng.gen::<f64>() > CROSSOVER_RATE {
            return (self.clone(), other.clone());
        }

        let mut child1 = self.clone();
        let mut child2 = other.clone();

        for i in 0..RGB_CHANNELS {
            for j in 0..GENE_LENGTH {
                if rng.gen_bool(0.5) {
                    child1.gene[i][j] = other.gene[i][j];
                    child2.gene[i][j] = self.gene[i][j];
                }
            }
        }

        (child1, child2)
    }

    fn get_val(&self) -> [u8; 3] {
        let mut vals = [0u8; 3];

        for (i, channel) in self.gene.iter().enumerate() {
            let mut val = 0u8;
            for &bit in channel {
                val = (val << 1) | if bit { 1 } else { 0 };
            }
            vals[i] = val;
        }

        vals
    }

    fn get_fitness(&self, target_image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> f64 {
        let target_pixel = target_image.get_pixel(self.pos.1 as u32, self.pos.0 as u32);
        let val = self.get_val();

        let mut diff_sum = 0.0;
        for i in 0..3 {
            let diff = val[i] as f64 - target_pixel[i] as f64;
            diff_sum += diff * diff;
        }
        
        let rmse = (diff_sum / 3.0).sqrt();
        let fitness = (-rmse / 50.0).exp();
        
        if rmse < 1.0 {
            fitness * 2.0
        } else {
            fitness
        }
    }
}

struct SimpleGA {
    #[allow(dead_code)]
    pos: (usize, usize),
    pool: Vec<Chromosome>,
}

impl SimpleGA {
    fn new(pos: (usize, usize)) -> Self {
        let pool = (0..POPULATION_SIZE)
            .map(|_| Chromosome::new(pos))
            .collect();

        SimpleGA { pos, pool }
    }

    fn tournament_selection(&self, target_image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> &Chromosome {
        let mut rng = thread_rng();

        let mut best = &self.pool[0];
        let mut best_fitness = best.get_fitness(target_image);

        for _ in 1..TOURNAMENT_SIZE {
            let candidate = &self.pool[rng.gen_range(0..self.pool.len())];
            let fitness = candidate.get_fitness(target_image);
            if fitness > best_fitness {
                best = candidate;
                best_fitness = fitness;
            }
        }

        best
    }

    fn get_fitness_stats(&self, target_image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> (f64, f64, f64) {
        let fitnesses: Vec<f64> = self.pool.iter()
            .map(|chr| chr.get_fitness(target_image))
            .collect();
        
        let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let max = fitnesses.iter().fold(0.0f64, |a, &b| a.max(b));
        let min = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        (avg, max, min)
    }

    fn step(&mut self, target_image: &ImageBuffer<Rgb<u8>, Vec<u8>>) {
        let mut new_pool = Vec::new();

        self.pool.sort_by(|a, b| {
            b.get_fitness(target_image)
                .partial_cmp(&a.get_fitness(target_image))
                .unwrap()
        });

        for i in 0..ELITE_SIZE.min(self.pool.len()) {
            new_pool.push(self.pool[i].clone());
        }

        while new_pool.len() < POPULATION_SIZE {
            let parent1 = self.tournament_selection(target_image);
            let parent2 = self.tournament_selection(target_image);

            let (mut child1, mut child2) = parent1.uniform_crossover(parent2);

            child1.mutate();
            child2.mutate();

            new_pool.push(child1);
            if new_pool.len() < POPULATION_SIZE {
                new_pool.push(child2);
            }
        }

        new_pool.truncate(POPULATION_SIZE);
        self.pool = new_pool;
    }

    fn get_best(&self, target_image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> &Chromosome {
        self.pool
            .iter()
            .max_by(|a, b| {
                a.get_fitness(target_image)
                    .partial_cmp(&b.get_fitness(target_image))
                    .unwrap()
            })
            .unwrap()
    }
}

fn load_target_image(path: &str) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let img_rgb = img.to_rgb8();
    Ok(image::imageops::resize(&img_rgb, IMG_SIZE as u32, IMG_SIZE as u32, image::imageops::FilterType::CatmullRom))
}

fn create_sample_image() -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (x * 255 / IMG_SIZE as u32) as u8;
        let g = (y * 255 / IMG_SIZE as u32) as u8;
        let b = ((x + y) * 255 / (IMG_SIZE as u32 * 2)) as u8;
        *pixel = Rgb([r, g, b]);
    }

    img
}

fn create_simple_gif_from_frames(frames: &[RgbImage], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    
    let mut palette = Vec::new();
    for r in 0..6 {
        for g in 0..6 {
            for b in 0..6 {
                palette.push((r * 51) as u8);
                palette.push((g * 51) as u8);
                palette.push((b * 51) as u8);
            }
        }
    }
    
    while palette.len() < 768 {
        palette.push(0);
    }

    let mut encoder = gif::Encoder::new(BufWriter::new(file), IMG_SIZE as u16, IMG_SIZE as u16, &palette)?;
    encoder.set_repeat(gif::Repeat::Infinite)?;

    let step = if frames.len() > 50 { frames.len() / 50 } else { 1 };

    for (i, frame) in frames.iter().enumerate() {
        if i % step != 0 {
            continue;
        }

        let mut indices = Vec::new();
        
        for pixel in frame.pixels() {
            let r = ((pixel[0] as f32 / 51.0).round() as usize).min(5);
            let g = ((pixel[1] as f32 / 51.0).round() as usize).min(5);
            let b = ((pixel[2] as f32 / 51.0).round() as usize).min(5);
            
            let index = r * 36 + g * 6 + b;
            indices.push(index as u8);
        }

        let mut gif_frame = gif::Frame::from_indexed_pixels(IMG_SIZE as u16, IMG_SIZE as u16, indices, None);
        gif_frame.delay = 20;
        encoder.write_frame(&gif_frame)?;
    }

    Ok(())
}

fn run_ga_with_output() {
    let target_image = match load_target_image("target.png") {
        Ok(img) => {
            println!("Target image loaded successfully");
            img
        }
        Err(_) => {
            println!("Could not load target.png, using generated sample image");
            create_sample_image()
        }
    };

    let mut ga_grid: Vec<Vec<SimpleGA>> = (0..IMG_SIZE)
        .map(|i| {
            (0..IMG_SIZE)
                .map(|j| SimpleGA::new((i, j)))
                .collect()
        })
        .collect();

    let mut frames = Vec::new();

    for gen in 0..ITERATION {
        println!("Generation {}/{}", gen + 1, ITERATION);

        for i in 0..IMG_SIZE {
            for j in 0..IMG_SIZE {
                ga_grid[i][j].step(&target_image);
            }
        }

        let mut frame = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);
        let mut total_fitness = 0.0;
        let mut perfect_matches = 0;
        
        for i in 0..IMG_SIZE {
            for j in 0..IMG_SIZE {
                let best = ga_grid[i][j].get_best(&target_image);
                let val = best.get_val();
                frame.put_pixel(j as u32, i as u32, Rgb([val[0], val[1], val[2]]));
                
                let fitness = best.get_fitness(&target_image);
                total_fitness += fitness;
                
                let target_pixel = target_image.get_pixel(j as u32, i as u32);
                if val[0] == target_pixel[0] && val[1] == target_pixel[1] && val[2] == target_pixel[2] {
                    perfect_matches += 1;
                }
            }
        }
        
        if gen % 25 == 0 || gen == ITERATION - 1 {
            let avg_fitness = total_fitness / (IMG_SIZE * IMG_SIZE) as f64;
            let match_percent = (perfect_matches as f64 / (IMG_SIZE * IMG_SIZE) as f64) * 100.0;
            println!("  Average fitness: {:.4}, Perfect matches: {:.2}% ({}/{})", 
                     avg_fitness, match_percent, perfect_matches, IMG_SIZE * IMG_SIZE);
            
            let (avg_fit, max_fit, min_fit) = ga_grid[IMG_SIZE/2][IMG_SIZE/2].get_fitness_stats(&target_image);
            println!("  Sample pixel fitness - Avg: {:.4}, Max: {:.4}, Min: {:.4}", avg_fit, max_fit, min_fit);
        }
        
        frames.push(frame);
    }

    if let Some(final_frame) = frames.last() {
        match final_frame.save("result.png") {
            Ok(_) => println!("Result saved as result.png"),
            Err(e) => println!("Failed to save result image: {}", e),
        }
    }

    match create_simple_gif_from_frames(&frames, "result.gif") {
        Ok(_) => println!("GIF saved as result.gif"),
        Err(e) => println!("Failed to create GIF: {}", e),
    }

    match target_image.save("target_sample.png") {
        Ok(_) => println!("Target image saved as target_sample.png"),
        Err(e) => println!("Failed to save target image: {}", e),
    }

    println!("GA process completed!");
}

fn main() {
    run_ga_with_output();
}