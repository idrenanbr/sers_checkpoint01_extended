# Checkpoint 01 – Data Science e Machine Learning no Python e Orange Data Mining

Este repositório contém a solução completa para as quatro partes do checkpoint de
ciência de dados e machine learning.  Ele inclui a análise da base
**Individual Household Electric Power Consumption** (tarefas 1–25 e 36–40) e do
**Appliances Energy Prediction** (tarefas 26–35).  As soluções são
apresentadas em um notebook Jupyter, em scripts Python e em arquivos
intermediários (figuras e CSVs) produzidos pela execução do código.

## Estrutura

- `analysis_complete.py` – script que executa todas as tarefas adicionais
  (21–40). Ao ser executado, ele gera gráficos e tabelas na pasta `outputs`.
- `notebooks/Checkpoint01_Complete.ipynb` – notebook que contém código e
  comentários explicando passo a passo as análises e modelos das
  quatro partes do checkpoint.
- `outputs/` – diretório com todos os arquivos gerados pelas análises: CSVs
  intermediários, gráficos (PNG), matriz de correlação, métricas de modelos,
  centros de clusters, etc.
- `energydata_complete.csv` – dataset de previsão de consumo de
  eletrodomésticos (obtido a partir da
  [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)).
- `hpc_data/household_power_consumption.txt` – dataset de consumo doméstico
  (obtido a partir da
  [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)).

## Como executar

1. Clone o repositório e navegue até a pasta raiz.
2. (Opcional) Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Certifique‑se de que os arquivos de dados `household_power_consumption.txt`
   e `energydata_complete.csv` estejam presentes nos caminhos indicados
   (`hpc_data/` e a raiz do repositório, respectivamente).  Caso o
   dataset de eletrodomésticos não esteja no repositório, ele pode ser
   baixado do UCI Machine Learning Repository e salvo com o nome
   `energydata_complete.csv`.

4. Execute o script para gerar os relatórios e gráficos:

   ```bash
   python analysis_complete.py --hpc-file hpc_data/household_power_consumption.txt \
       --appliances-file energydata_complete.csv --out outputs
   ```

5. Abra o notebook `notebooks/Checkpoint01_Complete.ipynb` em um ambiente
   Jupyter (Notebook ou Lab) e execute as células para acompanhar a
   análise passo a passo.

## Dependências

As principais bibliotecas utilizadas são:

- pandas
- numpy
- matplotlib
- scikit‑learn

As dependências completas estão listadas no arquivo `requirements.txt`.
