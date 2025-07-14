# Image Evolution GIF Generator (using Genetic Algorithm)

A genetic algorithm implementation that evolves a low-resolution image by attempting to approximate a target image through evolutionary computation.  
The program generates an animated GIF that visualizes the entire evolution process from random noise to a recognizable approximation using pixel-level mutations and selection.

## Demo
| **Original Image** | **Generated GIF** |
|:---:|:---:|
| ![target2](https://github.com/user-attachments/assets/40bb4ed8-588b-4f7f-96de-c58ba8bb0fed) | ![exa](https://github.com/user-attachments/assets/b3407c34-23cb-466a-89c1-54f476426efa) |

## Details
This project uses a genetic algorithm to evolve RGB pixel values for each position in a 100x100 image grid. 
Each pixel is represented by a chromosome containing binary genes for red, green, and blue color channels. 
Through selection, crossover, and mutation operations, the algorithm iteratively improves the image quality to match a target image. 

#### Genetic Algorithm Parameters
```bash
const IMG_SIZE: usize = 100;           // Image dimensions (100x100)
const POPULATION_SIZE: usize = 6;      // Population size per pixel
const ITERATION: usize = 50;           // Number of generations
const MUTATION_RATE: f64 = 0.05;       // Bit-flip mutation probability
const CROSSOVER_RATE: f64 = 0.8;       // Crossover probability
const GENE_LENGTH: usize = 8;          // Bits per color channel
const RGB_CHANNELS: usize = 3;         // Red, Green, Blue channels
const TOURNAMENT_SIZE: usize = 3;      // Tournament selection size
const ELITE_SIZE: usize = 2;           // Number of elite individuals preserved
```
For more details, see [here](https://github.com/Yutarop/ga-pixel-art/wiki).

## Usage
#### Prerequisites
Add these dependencies to your Cargo.toml:
```bash
[dependencies]
image = "0.24"
rand = "0.8"
gif = "0.12"
```
#### Running the Program
```bash
# Place a target image named target.png in the project directory.
cargo run
```

#### Output Files
- result.png: Final evolved image
- result.gif: Animated evolution process
- target_sample.png: Copy of the target image used
