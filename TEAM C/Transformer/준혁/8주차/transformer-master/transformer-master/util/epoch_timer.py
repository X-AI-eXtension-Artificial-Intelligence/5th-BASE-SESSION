# 에포크 시간 계산 함수
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time  # 경과 시간 계산
    elapsed_mins = int(elapsed_time / 60)  # 경과 시간을 분으로 변환
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # 남은 초 계산
    return elapsed_mins, elapsed_secs  # 분과 초 반환