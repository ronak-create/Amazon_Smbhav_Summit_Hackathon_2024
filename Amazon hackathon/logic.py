# Import libraries
import pandas as pd
import openai
from transformers import pipeline
from openai import OpenAI

# Set up OpenAI API key
openai.api_key = "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"  # Replace with your OpenAI API key
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)


# Initialize Hugging Face pipeline for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample dataset for demonstration
data = {
    "Product Name": ["Mobile Phones", "Organic Spices", "Pharmaceuticals"],
    "Product Type": ["Electronics", "Agriculture", "Medicine"],
    "Country": ["India", "US", "France"],
    "Regulations": [
        "BIS Certification, WPC Approval, Labeling Standards",
        "USDA Organic Certification, FSMA, GMP",
        "EMA Approval, Clinical Trials Authorization, EU Directive"
    ],
    "Attributes": [
        "Lithium Battery, Wireless Communication",
        "No Pesticides, Organic",
        "Liquid Medicine, Prescription Required"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define function to summarize regulations
def summarize_regulations(text):
    try:
        summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Define function to interact with ChatGPT for dynamic suggestions
def suggest_actions_with_gpt(product_name, product_type, country, attributes):
    try:
        # Create a structured prompt
        prompt = f"""
        Product Name: {product_name}
        Product Type: {product_type}
        Country: {country}
        Attributes: {attributes}
        Task: Provide specific regulations, certifications, and actionable steps for compliance in {country}.
        """
        
        # Query GPT
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract and return the GPT response
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error with GPT: {str(e)}"

# Process the dataset
df["Regulatory Summary"] = df["Regulations"].apply(summarize_regulations)
df["Suggested Actions"] = df.apply(
    lambda row: suggest_actions_with_gpt(
        row["Product Name"], row["Product Type"], row["Country"], row["Attributes"]
    ), axis=1
)

# Save the dataset to a CSV file
output_file = "regulatory_summary_and_suggestions_with_gpt.csv"
df.to_csv(output_file, index=False)

# Display the processed dataset
print("Summarized and Suggested Data with GPT Integration:")
print(df)

# Notify file saved
print(f"\nData saved to {output_file}")