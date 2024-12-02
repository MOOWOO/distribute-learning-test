#!/bin/bash

# ========== 환경 변수 설정 ==========
# DeepSpeed 및 NCCL 디버깅 관련 환경 변수 설정
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export PYTHONUNBUFFERED=1

# ======= 노드 및 슬롯 정보 =======
# 호스트 파일 이름
HOSTFILE="hostfile"

# ======= SSH 테스트 =======
echo "노드 간 SSH 연결 테스트..."
while IFS= read -r NODE; do
    HOST=$(echo $NODE | awk '{print $1}')
    echo "테스트: $HOST"
    ssh -o BatchMode=yes -o ConnectTimeout=5 work@$HOST "echo SSH 연결 성공: $HOST" || { 
        echo "❌ $HOST에 SSH 연결 실패. 확인 필요."; exit 1; 
    }
done < $HOSTFILE

echo "모든 노드에 SSH 연결 성공 🎉"

# ======= 모든 노드에서 환경 준비 =======
echo "모든 노드에서 환경 준비 중..."
while IFS= read -r NODE; do
    HOST=$(echo $NODE | awk '{print $1}')
    ssh work@$HOST << EOF
        echo "[$(hostname)] Python 및 DeepSpeed 환경 확인 중..."
        which python || echo "❌ Python 설치 필요"
        pip show deepspeed || echo "❌ DeepSpeed 설치 필요"
EOF
done < $HOSTFILE

# ======= DeepSpeed 실행 =======
echo "DeepSpeed 실행..."
deepspeed --hostfile=$HOSTFILE train_llm.py
