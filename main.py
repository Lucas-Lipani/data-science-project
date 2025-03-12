import pandas as pd
import os

def main():
    # Listar todos os arquivos na pasta
    arquivos = os.listdir("data")

    # Filtrar apenas arquivos CSV
    arquivos_csv = [arquivo for arquivo in arquivos if arquivo.endswith(".csv")]

    # # Percorrer cada arquivo CSV
    # for arquivo_csv in arquivos_csv:
    #     caminho_completo = os.path.join("data", arquivo_csv)
    #     print(f"Processando arquivo: {caminho_completo}")
        
    #     # Ler o arquivo CSV
    #     df = pd.read_csv(caminho_completo)
        
    #     # Exemplo: Exibir as colunas do DataFrame
    #     print(df.columns)

    # Carregar o arquivo CSV
    df = pd.read_csv("data/players.csv")

    # Filtrar jogadores brasileiros
    jogadores_brasileiros = df[df["country_of_citizenship"] == "Brazil"]
    jogadores_brasileiros_ordenados = jogadores_brasileiros.sort_values(by="market_value_in_eur", ascending=False)

    # Exibir os jogadores brasileiros
    print(jogadores_brasileiros_ordenados)

if __name__ == "__main__":
    main() 
# %run -i -n main.py
