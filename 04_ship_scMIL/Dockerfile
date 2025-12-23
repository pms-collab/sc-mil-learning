FROM mambaorg/micromamba:1.5.10

WORKDIR /work

# 1) env spec 먼저 복사(캐시 효율)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /work/environment.yml

# 2) env 생성
RUN micromamba create -y -n scmil -f /work/environment.yml && \
    micromamba clean -a -y

# 3) 컨테이너 실행 시 scmil 자동 활성화
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PYTHONUNBUFFERED=1

# 4) 코드 복사 (옵션: run 시 바인드 마운트로 덮어씀)
COPY --chown=$MAMBA_USER:$MAMBA_USER . /work

CMD ["bash"]
