FROM nginx:stable-alpine@sha256:47928bf67eb62a1863cb08d9ecb52d3e704304ba614eaee17884204677702951
ENV BACKEND_URL=http://counter:8080 \
    API_BASE_URL=""
COPY public/ /usr/share/nginx/html/
COPY nginx.conf.template /etc/nginx/countwell.conf.template
COPY entrypoint.sh /docker-entrypoint.d/40-countwell-config.sh
RUN chmod +x /docker-entrypoint.d/40-countwell-config.sh
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz
