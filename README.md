<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Projeto de Classificação de preços de residências</h1>
    <p>Este projeto utiliza um modelo de aprendizado de máquina para realizar classificações. O modelo foi salvo no formato <code>.pkl</code> e pode ser carregado para realizar previsões.</p>
    <h2>Instalação</h2>
    <p>Antes de utilizar o projeto, é necessário instalar os pacotes listados no arquivo <code>requirements.txt</code>. Para isso, execute o seguinte comando no terminal:</p>
    <pre><code>pip install -r requirements.txt</code></pre>
    <h2>Como Usar o Modelo</h2>
    <p>Para carregar o modelo salvo e utilizá-lo para fazer previsões, use o seguinte código em Python:</p>
    <pre><code>import pickle
import numpy as np

# Carregar o modelo salvo
with open('randomforestregressor.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Criar uma amostra de entrada para previsão
amostra = {id: xx, host_id: xx, ...}  # Substitua pelos seus dados reais

# Fazer a previsão
resultado = modelo.predict(amostra)
</body>
</html>
