use std::ops::Index;

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

pub trait Individual {
    fn create(chromosome: Chromosome) -> Self;

    fn fitness(&self) -> f32;

    fn chromosome(&self) -> &Chromosome;
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_chromosome() -> Chromosome {
        Chromosome {
            genes: vec![3.0, 1.0, 2.0],
        }
    }

    #[test]
    fn len() {
        assert_eq!(make_chromosome().len(), 3);
    }

    #[test]
    fn iter() {
        let chromosome = make_chromosome();
        let genes: Vec<_> = chromosome.iter().collect();

        assert_eq!(genes.len(), 3);
        assert_eq!(genes[0], &3.0);
        assert_eq!(genes[1], &1.0);
        assert_eq!(genes[2], &2.0);
    }

    #[test]
    fn iter_mut() {
        let mut chromosome = make_chromosome();

        chromosome.iter_mut().for_each(|gene| {
            *gene *= 10.0;
        });

        let genes: Vec<_> = chromosome.iter().collect();

        assert_eq!(genes.len(), 3);
        assert_eq!(genes[0], &30.0);
        assert_eq!(genes[1], &10.0);
        assert_eq!(genes[2], &20.0);
    }

    #[test]
    fn from_iterator() {
        let chromosome: Chromosome = vec![3.0, 1.0, 2.0].into_iter().collect();

        assert_eq!(chromosome[0], 3.0);
        assert_eq!(chromosome[1], 1.0);
        assert_eq!(chromosome[2], 2.0);
    }

    #[test]
    fn index() {
        let chromosome = Chromosome {
            genes: vec![3.0, 1.0, 2.0],
        };

        assert_eq!(chromosome[0], 3.0);
        assert_eq!(chromosome[1], 1.0);
        assert_eq!(chromosome[2], 2.0);
    }

    #[test]
    fn into_iterator() {
        let chromosome = Chromosome {
            genes: vec![3.0, 1.0, 2.0],
        };

        let genes: Vec<_> = chromosome.into_iter().collect();

        assert_eq!(genes.len(), 3);
        assert_eq!(genes[0], 3.0);
        assert_eq!(genes[1], 1.0);
        assert_eq!(genes[2], 2.0);
    }

    mod histogram {
        use crate::{selection::{RouletteWheelSelection, SelectionMethod, UniformCrossover}, chromosome::Chromosome, GeneticAlgorithm, mutation::GaussianMutation};

        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use std::collections::BTreeMap;

        #[derive(Clone, Debug, PartialEq)]
        pub enum TestIndividual {
            /// For tests that require access to chromosome
            WithChromosome { chromosome: Chromosome },

            /// For tests that don't require access to chromosome
            WithFitness { fitness: f32 },
        }

        impl TestIndividual {
            pub fn new(fitness: f32) -> Self {
                Self::WithFitness { fitness }
            }
        }

        impl PartialEq for Chromosome {
            fn eq(&self, other: &Self) -> bool {
                approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice(),)
            }
        }

        impl Individual for TestIndividual {
            fn create(chromosome: Chromosome) -> Self {
                Self::WithChromosome { chromosome }
            }

            fn chromosome(&self) -> &Chromosome {
                match self {
                    Self::WithChromosome { chromosome } => chromosome,
                    Self::WithFitness { .. } => {
                        panic!("Not supported for TestIndividual::WithFitness")
                    }
                }
            }

            fn fitness(&self) -> f32 {
                match self {
                    Self::WithChromosome { chromosome } => chromosome.iter().sum(),

                    Self::WithFitness { fitness } => *fitness,
                }
            }
        }

        fn individual(genes: &[f32]) -> TestIndividual {
            let chromosome = genes.iter().cloned().collect();

            TestIndividual::create(chromosome)
        }

        #[test]
        fn test() {
            let method = RouletteWheelSelection::new();
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let population = vec![
                TestIndividual::new(2.0),
                TestIndividual::new(1.0),
                TestIndividual::new(4.0),
                TestIndividual::new(3.0),
            ];

            let actual_histogram: BTreeMap<i32, _> = (0..1000)
                .map(|_| method.select(&mut rng, &population))
                .fold(Default::default(), |mut histogram, individual| {
                    *histogram.entry(individual.fitness() as _).or_default() += 1;

                    histogram
                });

            let expected_histogram = maplit::btreemap! {
                // fitness => how many times this fitness has been chosen
                1 => 98,
                2 => 202,
                3 => 278,
                4 => 422,
            };

            assert_eq!(expected_histogram, actual_histogram);
        }

        #[test]
        fn test2() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let ga = GeneticAlgorithm::new(
                RouletteWheelSelection::new(),
                UniformCrossover,
                GaussianMutation::new(0.5, 0.5),
            );

            let mut population = vec![
                individual(&[0.0, 0.0, 0.0]), // fitness = 0.0
                individual(&[1.0, 1.0, 1.0]), // fitness = 3.0
                individual(&[1.0, 2.0, 1.0]), // fitness = 4.0
                individual(&[1.0, 2.0, 4.0]), // fitness = 7.0
            ];

            // We're running `.evolve()` a few times, so that the
            // differences between initial and output population are
            // easier to spot.
            //
            // No particular reason for a number of 10 - this test would
            // be fine for 5, 20 or even 1000 generations; the only thing
            // that'd change is the *magnitude* of difference between
            // initial and output population.
            for _ in 0..10 {
                population = ga.evolve(&mut rng, &population);
            }

            let expected_population = vec![
                individual(&[0.44769490, 2.0648358, 4.3058133]), // fitness ~= 6.8
                individual(&[1.21268670, 1.5538777, 2.8869110]), // fitness ~= 5.7
                individual(&[1.06176780, 2.2657390, 4.4287640]), // fitness ~= 7.8
                individual(&[0.95909685, 2.4618788, 4.0247330]), // fitness ~= 7.4
            ];

            assert_eq!(expected_population, population);
        }
    }
}
