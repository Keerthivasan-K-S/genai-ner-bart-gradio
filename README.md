## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

## AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

## PROBLEM STATEMENT:
The goal is to develop an application that can accurately recognize and categorize named entities such as persons, organizations, locations, dates, etc., from input text. By fine-tuning a pre-trained BART model specifically for NER tasks, the system should be able to understand contextual relationships and identify relevant entities. The Gradio framework will be used to build a user-friendly interface for real-time interaction and evaluation.

## DESIGN STEPS:

### STEP 1:Data Preparation and Model Fine-Tuning

Collect or obtain a labeled NER dataset (e.g., CoNLL-2003). Preprocess the dataset for tokenization and entity tagging. Fine-tune a pre-trained BART model on the dataset using Hugging Face's transformers library.

### STEP 2:Model Evaluation

Test the fine-tuned BART model on a validation set. Evaluate performance using standard metrics like F1-score, precision, and recall. Perform necessary adjustments and re-training if needed.

### STEP 3:Gradio Interface Development

Use Gradio to build a simple interface where users can input text. Integrate the fine-tuned model to process the input and display recognized entities in real-time. Ensure the interface is intuitive and allows users to submit queries for NER.

## PROGRAM:

```py
from transformers import pipeline
import gradio as gr

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

def recognize_entities(text):

    entities = ner_pipeline(text)

    if not entities:
        return "No entities found."

    result = ""

    for e in entities:
        result += (
            f"Entity: {e['word']}\n"
            f"Type: {e['entity_group']}\n"
            f"Confidence: {round(e['score'],4)}\n\n"
        )

    return result

interface = gr.Interface(
    fn=recognize_entities,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter text..."
    ),
    outputs="text",
    title="Named Entity Recognition Prototype",
    description="NER using Hugging Face Transformer Model and Gradio"
)

interface.launch()
```

## OUTPUT:
<img width="1825" height="856" alt="image" src="https://github.com/user-attachments/assets/5f185b63-8180-409b-a5b7-a751cbb08af4" />


## RESULT:
Thus, the project resulted in a functional NER prototype using a fine-tuned BART model and Gradio for real-time entity recognition, showcasing strong performance and user-friendly deployment.
