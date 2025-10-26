# gsim

This project is a quantum circuit simulator based on the paper [Lie-algebraic classical simulations for quantum computing](https://arxiv.org/abs/2308.01432).

## Installation

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
gsim = { git = "https://github.com/2463/gsim.git" }
```

## Usage

Here is a basic example of how to use the `gsim` crate:

```rust
use gsim::{make_gsim, run_simulation};
use nalgebra::DMatrix;
use num::complex::Complex64;

fn main() {
    // Define the initial density matrix, observable, and Hamiltonians
    let init_density_matrix = DMatrix::<Complex64>::identity(2, 2);
    let observable = DMatrix::<Complex64>::identity(2, 2);
    let hamiltonians = vec![DMatrix::<Complex64>::identity(2, 2)];

    // Create a new GSim instance
    let mut gsim = make_gsim(init_density_matrix, observable, hamiltonians);

    // Define the parameters and gate numbers
    let params_and_gate_numbers = vec![(1.0, 0)];

    // Run the simulation
    let expectation_value = run_simulation(&mut gsim, params_and_gate_numbers);

    println!("Expectation value: {}", expectation_value);
}
```

## Contributing

Pull requests are welcome.

## License

This project is licensed under the MIT License.
