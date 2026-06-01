# Basket option N-D package

Este pacote adiciona basket option aritmética N-dimensional ao projeto.

## O que foi adicionado

- `equation/basket_option.py`: problema de basket option N-D com benchmark Gauss-Hermite cacheado.
- `optimize/basket_option.py`: otimizador PINN para PDE multivariada com termos cruzados de correlação.
- `method/nn_nd.py`: MLP/ResNet flexíveis para entrada N+1.
- `basket_configs.py`: parâmetros realistas de sigma/rho e sweeps.
- `basket_experiment_utils.py`: funções de treino/salvamento no padrão do projeto.
- Scripts separados por dimensão e modelo:
  - `generate_basket_2d_classic.py`
  - `generate_basket_2d_qnn.py`
  - `generate_basket_2d_hqnn.py`
  - `generate_basket_3d_classic.py`
  - `generate_basket_3d_qnn.py`
  - `generate_basket_3d_hqnn.py`
- Scripts para rodar tudo:
  - `run_basket_2d_all.py`
  - `run_basket_3d_all.py`
  - `run_all_basket.py`
- `calculate_basket_greeks_from_runs.py`: calcula preço, Delta_i, Gamma_ii, Gamma_ij e Theta.

## Benchmark Gauss-Hermite

O benchmark é calculado uma vez e salvo em:

```text
experimentos_pinn_basket/benchmarks/
```

As Greeks de benchmark também são cacheadas em:

```text
experimentos_pinn_basket/basket_greeks/benchmark_cache/
```

## Uso

Na raiz do projeto:

```bash
python generate_basket_2d_classic.py
python generate_basket_2d_qnn.py
python generate_basket_2d_hqnn.py
```

ou:

```bash
python run_basket_2d_all.py
python run_basket_3d_all.py
```

ou tudo:

```bash
python run_all_basket.py
```

Depois:

```bash
python calculate_basket_greeks_from_runs.py
```

## Configurações realistas iniciais

2D:

```python
sigmas = [0.20, 0.25]
rho = [[1.00, 0.50], [0.50, 1.00]]
```

3D:

```python
sigmas = [0.20, 0.25, 0.30]
rho = [[1.00, 0.50, 0.50], [0.50, 1.00, 0.50], [0.50, 0.50, 1.00]]
```

Para mudar, edite `basket_configs.py`.
