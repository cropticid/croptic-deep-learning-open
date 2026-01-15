# test_modal.py - VERSI BENAR
import modal

# Load deployed function (app name: "croptic-deeplearning", function name: "run_pipeline")
run_pipeline = modal.Function.from_name("croptic-deeplearning", "run_pipeline")

# Cara 1: .remote() - synchronous (wait hasil)
#result = run_pipeline.remote(user_id="123")
#print(result)

# Cara 2: .spawn() - asynchronous (fire & forget, dapat FunctionCall)
fc = run_pipeline.spawn(user_id="123")
result = fc.get()  # wait hasil kalau perlu
print(result)
