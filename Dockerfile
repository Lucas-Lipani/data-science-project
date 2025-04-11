# Usa uma imagem Python oficial
FROM python:3.10

# Define o diretório de trabalho no container
WORKDIR /app

# Copia todos os arquivos do projeto para dentro do container
COPY . /app

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta que o Flask usará
EXPOSE 5000


# Comando para rodar o servidor Flask
CMD ["python", "backend/app.py"]
