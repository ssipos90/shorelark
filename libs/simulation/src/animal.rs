use lib_neural_network as nn;
use nalgebra as na;
use rand::{Rng, RngCore};

use crate::Eye;

#[derive(Debug)]
pub struct Animal {
    pub(crate) position: na::Point2<f32>,
    pub(crate) rotation: na::Rotation2<f32>,
    pub(crate) speed: f32,
    pub(crate) eye: Eye,
    pub(crate) brain: nn::Network,
    pub(crate) satiation: usize,
}

impl Animal {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let eye = Eye::default();

        let brain = nn::Network::random(
            rng,
            &[
                // The Input Layer
                //
                // Because our eye returns Vec<f32>, and our neural
                // network works on Vec<f32>, we can pass-through
                // numbers from eye into the neural network directly.
                //
                // Had our birdies had, I dunno, ears, we could do
                // something like: `eye.cells() + ear.nerves()` etc.
                nn::LayerTopology {
                    neurons: eye.cells(),
                },
                // The Hidden Layer
                //
                // There is no best answer as to "how many neurons
                // the hidden layer should contain" (or how many
                // hidden layers there should be, even - there could
                // be zero, one, two or more!).
                //
                // The rule of thumb is to start with a single hidden
                // layer that has somewhat more neurons that the input
                // layer, and see how well the network performs.
                nn::LayerTopology {
                    neurons: 2 * eye.cells(),
                },
                // The Output Layer
                //
                // Since the brain will control our bird's speed and
                // rotation, this gives us two numbers = two neurons.
                nn::LayerTopology { neurons: 2 },
            ],
        );

        Self {
            position: rng.gen(),
            // ------ ^-------^
            // | If not for `rand-no-std`, we'd have to do awkward
            // | `na::Point2::new(rng.gen(), rng.gen())` instead
            // ---
            rotation: rng.gen(),
            speed: 0.002,
            eye,
            brain,
            satiation: 0,
        }
    }

    pub fn position(&self) -> na::Point2<f32> {
        // ------------------ ^
        // | No need to return a reference, because na::Point2 is Copy.
        // |
        // | (meaning: it's so small that cloning it is cheaper than
        // | messing with references.)
        // |
        // | Of course you don't have to memorize which types are Copy
        // | and which aren't - if you accidentally return a reference
        // | to a type that's Copy, rust-clippy will point it out and
        // | suggest a change :-)
        // ---

        self.position
    }

    pub fn rotation(&self) -> na::Rotation2<f32> {
        self.rotation
    }
}
