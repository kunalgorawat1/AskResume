import ollama

response = ollama.list()

# # print(response)

# res = ollama.chat(
#     model="llama3.2",
#     messages=[
#         {
#             "role": "user",
#             "content": "Tell me about space time and black holes."
#         }
#     ]
# )
# print(res['message']['content'])

# Create e new model with modelfile
modelFile1 = """
FROM llama3.2
SYSTEM You are an intelligent assistant who knows everything about football and answer questions with brevity. 
PARAMETER temperature 0.1
"""
# help(ollama.create)

ollama.create(model="football_agent", system=modelFile1, from_="llama3.2")

res = ollama.generate(model="football_agent", prompt="Who is the greatest football player of all time?")

print(res)