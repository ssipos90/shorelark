/********************
*  Network
*********************/
use rand::{Rng, RngCore};

pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    #[cfg(test)]
    fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    // pub fn from_weights(layers: &[LayerTopology], weights: impl IntoIterator<Item = 32>) -> Self {
    //     todo!()
    // }

    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        Self {
            layers: layers
                .windows(2)
                .map(|layers| {
                    let input_neurons = layers[0].neurons;
                    let output_neurons = layers[1].neurons;

                    Layer::random(rng, input_neurons, output_neurons)
                })
                .collect(),
        }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }

    pub fn weights(&self) -> Vec<f32> {
        use std::iter::once;
        self.layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .copied()
            .collect()
    }
}

/********************
*  Layer
*********************/

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    #[cfg(test)]
    fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        Self {
            neurons: (0..output_neurons)
                .map(|_| Neuron::random(rng, input_neurons))
                .collect(),
        }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}

/********************
*  Neuron
*********************/
#[derive(Debug)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    #[cfg(test)]
    fn new(bias: f32, weights: Vec<f32>) -> Self {
        Self { bias, weights }
    }

    fn random(rng: &mut dyn RngCore, input_neurons: usize) -> Self {
        Self {
            bias: rng.gen_range(-1.0..=1.0),
            weights: (0..input_neurons)
                .map(|_| rng.gen_range(-1.0..=1.0))
                .collect(),
        }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        // TODO: use Result instead of assert
        assert_eq!(inputs.len(), self.weights.len());

        inputs
            .iter()
            .zip(&self.weights)
            .fold(self.bias, |acc, (input, weight)| acc + input * weight)
            .max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    mod network {
        use super::*;
        #[test]
        fn test_layer_count() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let network = Network::random(
                &mut rng,
                &[
                    LayerTopology { neurons: 1 },
                    LayerTopology { neurons: 2 },
                    LayerTopology { neurons: 3 },
                ],
            );
            assert_eq!(network.layers.len(), 2);
        }

        #[test]
        fn test_weigths() {
            let network = Network::new(vec![
                Layer::new(vec![Neuron::new(0.1, vec![0.2, 0.3, 0.4])]),
                Layer::new(vec![Neuron::new(0.5, vec![0.6, 0.7, 0.8])]),
            ]);
            let actual = network.weights();
            let expected = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
        }
    }

    mod layer {
        use super::*;
        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 3, 2);

            approx::assert_relative_eq!(layer.neurons[0].bias, -0.6255188);
            approx::assert_relative_eq!(
                layer.neurons[0].weights.as_slice(),
                [0.67383957, 0.8181262, 0.26284897].as_ref(),
            );
            approx::assert_relative_eq!(layer.neurons[1].bias, 0.5238807);
            approx::assert_relative_eq!(
                layer.neurons[1].weights.as_slice(),
                [-0.53516835, 0.069369674, -0.7648182].as_ref(),
            );
        }
    }

    mod randomnes {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test() {
            // Because we always use the same seed, our `rng` in here will
            // always return the same set of values
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random(&mut rng, 4);

            approx::assert_relative_eq!(-0.6255188, neuron.bias);
            approx::assert_relative_eq!(
                neuron.weights.as_slice(),
                [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref(),
            );
        }
    }

    mod propagate {
        use super::*;
        #[test]
        fn test() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };

            // Ensures `.max()` (our ReLU) works:
            approx::assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

            // `0.5` and `1.0` chosen by a fair dice roll:
            approx::assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
            );

            // We could've written `1.15` right away, but showing the entire
            // formula makes our intentions clearer
        }
    }
}
