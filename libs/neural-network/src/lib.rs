/********************
*  Network
*********************/
use rand::{Rng, RngCore};

pub struct LayerTopology {
    pub neurons: usize,
}

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn random<R: RngCore>(rng: &mut R, layers: &[LayerTopology]) -> Self {
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

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

/********************
*  Layer
*********************/

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn random<R: RngCore>(rng: &mut R, input_neurons: usize, output_neurons: usize) -> Self {
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
    fn random<R: RngCore>(rng: &mut R, input_neurons: usize) -> Self {
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
            assert_eq!(3, network.layers.len());
        }
    }

    mod layer {
        use super::*;
        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 3, 2);

            approx::assert_relative_eq!(-0.6255188, layer.neurons[0].bias);
            approx::assert_relative_eq!(
                [0.67383957, 0.8181262, 0.26284897].as_ref(),
                layer.neurons[0].weights.as_slice()
            );
            approx::assert_relative_eq!(0.5238807, layer.neurons[1].bias);
            approx::assert_relative_eq!(
                [-0.53516835, 0.069369674, -0.7648182].as_ref(),
                layer.neurons[1].weights.as_slice()
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

            approx::assert_relative_eq!(neuron.bias, -0.6255188);
            approx::assert_relative_eq!(
                [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref(),
                neuron.weights.as_slice()
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
