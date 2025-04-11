# Usa uma imagem Nginx pronta
FROM nginx:alpine

# Remove a página padrão do Nginx
RUN rm -rf /usr/share/nginx/html/*

# Copia seus arquivos HTML/CSS/JS para o Nginx
COPY . /usr/share/nginx/html

# Expõe a porta 80 (web)
EXPOSE 80
