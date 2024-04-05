def check_packages():
    try:
        import langchain
        _has_packages = True
    except (ImportError, AttributeError):
        _has_packages = False

    if _has_packages:
        print("Proceed.")
    else:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("[ERROR] 0번 모듈 노트북(0_setup.ipynb)을 먼저 실행해 주세요.")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")