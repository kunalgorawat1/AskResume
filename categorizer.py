import ollama
import os

model = "llama3.2"

# paths to input and outpur
input_file = "./data/grocery_list.txt"
output_file = "./data/categorized_grocery_list.txt"

# ensue input exists
if not os.path.exists(input_file):
    print(f"Input file '{input_file}' not found.")
    exit(1)


# read the normal file 
with open(input_file, "r") as f:
    items = f.read().strip()

# promt
prompt = f"""
You are an assistant that categorizes and sorts grocery items.
Here is a list of grocery items:
{items}

TODO:
1. Categorize these items into appropriate categories such as dairy, beverage, etc.
2. sort the item alphabatically within each category.
3. Write a sort nutrition info next to the items per 100g of the item like item(protein: 10g, carbs: 2g)
4. Also wirte the healyh benefits of the items after the nutrition info.
5. Keep these details in one line per item and present in a clear and organized manner.
"""

try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print(generated_text)

    # write the categorized list to output file
    with open(output_file, "w") as f:
        f.write(generated_text.strip())

except Exception as e:
    print("An error occured: ", str(e))