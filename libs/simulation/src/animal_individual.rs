use crate::*;

pub struct AnimalIndividual {
    fitness: f32,
    chromosome: ga::Chromosome,
}

impl AnimalIndividual {
    pub fn from_animal(animal: &Animal) -> Self {
        Self {
            chromosome: animal.brain.weights(),
            fitness: animal.satiation as f32,
        }
    }

    pub fn into_animal(&self, rng: &mut dyn RngCore) -> Animal {
        todo!();
    }
}

impl ga::Individual for AnimalIndividual {
    fn create(chromosome: ga::Chromosome) -> Self {
        todo!()
    }

    fn fitness(&self) -> f32 {
        self.fitness
    }

    fn chromosome(&self) -> &ga::Chromosome {
        &self.chromosome
    }
}
