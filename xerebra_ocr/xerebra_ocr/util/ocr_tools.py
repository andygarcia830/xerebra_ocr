import frappe
from pdf2image import convert_from_path
import numpy as np
import warnings
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import json
from PIL import Image
import fitz
from google.cloud import vision
import io

from frappe.utils import get_site_name

warnings.filterwarnings("ignore")

def authenticate():
    settings = frappe.get_doc('Xerebra OCR Settings')
    saj = f'{get_site_name(frappe.local.request.host)}/{settings.google_cloud_service_account_json}'
    os.environ['OPENAI_API_KEY'] = settings.openai_api_key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saj


def save_rgba_as_jpeg(img, byte_stream):
    """Saves an RGBA image as JPEG by converting to RGB."""

    try:
        if img.mode == 'RGBA':
            rgb_img = img.convert('RGB')  # Convert RGBA to RGB
            rgb_img.save(byte_stream, 'JPEG')
            print(f"Image saved as JPEG to {byte_stream}")
        else:
            img.save(byte_stream, 'JPEG') #if it is already RGB, save it.
            print(f"Image saved as JPEG to {byte_stream}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def detect_text_google_vision(pil_image):
    """Performs text detection on a local image file.

    Args:
        file_path: The path to the local image file.

    Returns:
        A list of dictionaries, where each dictionary contains the text and its bounding box vertices.
    """

    client = vision.ImageAnnotatorClient()

    # with io.open(file_path, 'rb') as image_file:
    #     content = image_file.read()
    with io.BytesIO() as byte_stream:
        # pil_image.save(byte_stream, format='JPEG')  # Convert PIL Image to JPEG
        save_rgba_as_jpeg(pil_image, byte_stream)
        image_content = byte_stream.getvalue()
        image = vision.Image(content=image_content)

        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        results = []
        for text in texts[1:]:  # Skip the first element, which is the full text
            vertices = [[vertex.x, vertex.y] for vertex in text.bounding_poly.vertices]
            # vertices = [vertices[0], vertices[1], vertices[3], vertices[2]]
            # results.append({"text": text.description, "bounding_box": vertices,"confidence": text.confidence,})
            results.append((vertices, text.description, text.confidence))

        return results

def is_image_file(file_path):
    """Check if a file is an image using Pillow."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify if it’s an image
        return True
    except Exception:
        return False
    

def is_image_pdf(file_path):
    """Check if a PDF contains images."""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            if page.get_images():
                return True
        return False
    except Exception as e:
        print(f"Error checking PDF: {e}")
        return False


def process_result(data, row_threshold =75, column_thresold = 500, confidence_threshold = 0.10):
    exploded = []
    confidence_list = []
    for item in data:
        entry=[]
        bb = item[0]
        entry.append(bb[0])
        entry.append(bb[1])
        entry.append(bb[2])
        entry.append(bb[3])
        entry.append(item[1]) # Text
        entry.append(item[2]) # Confidence
        # entry.append(item[2])
        exploded.append(entry)
    df = pd.DataFrame(exploded, columns=['Left-Top', 'Right-Top', 'Left-Bottom', 'Right-Bottom', 'Text', 'Confidence'])
    # print(df[['Top-Left','Text']])

    # Extract y-coordinate to determine rows and x-coordinate for columns
    df['Left'] = df['Left-Top'].apply(lambda x: x[0])
    df['Top'] = df['Left-Top'].apply(lambda x: x[1])
    df['Right'] = df['Right-Bottom'].apply(lambda x: x[0])
    df['Bottom'] = df['Right-Bottom'].apply(lambda x: x[1])
    

    # Sorting by the top and left coordinates to approximate visual arrangement
    df = df.sort_values(by=['Top', 'Left'])

    # Threshold for grouping texts in the same row
    
    # Function to assign row numbers
    def assign_row_number(y_coord):
        if y_coord < row_threshold:
            return 0
        return (y_coord // row_threshold)

    df['Row'] = df['Top'].apply(assign_row_number)

    def process_row(x):
        x = x.sort_values(by = 'Left')
        # print(f'ENTRY')
        text_row=[]
        # print(f'SHAPE {x.shape}')
        first_token = True
        this_text = ''
        for ctr, token in x.reset_index(drop='True').iterrows():
            # print(f'CTR {ctr} TOKEN {token}')
            if token['Confidence'] > confidence_threshold:
                confidence_list.append(token['Confidence'])
                if first_token:
                    this_text = token["Text"]
                    first_token = False
                else:
                    horiz_space = token['Left'] - x.iloc[ctr -1]['Right']
                    if horiz_space > column_thresold:
                        text_row.append(this_text)
                        this_text = token["Text"]
                    else:
                        this_text = ' '.join([this_text, token["Text"]])
                        
                    # print(f'CONT TOKEN {token["Text"]} {horiz_space}')
            if ctr == x.shape[0] -1 and len(this_text) > 0:
                text_row.append(this_text)
        # print(f'ROW {text_row}')
        return text_row

        # return x
    # Group by row and concatenate text with some spacing
    grouped_text = df[['Row','Top','Left','Bottom','Right','Text','Confidence']].groupby('Row').apply(process_row)

    # Resulting grouped text as a simpler representation of the table
    # print(f'GROUPED TEXT {grouped_text.values}')
    return grouped_text.values, np.mean(confidence_list)


def ocr(pages, no_pages, row_threshold =10, column_thresold = 500, confidence_threshold = 0.10, check_rotation = False):
    result = []
    for i, pil_image in enumerate(pages):
        if i ==0 or i ==len(pages) -1:
            results = detect_text_google_vision(pil_image)

            print(results)
            
            processed, confidence = process_result(results,  row_threshold =row_threshold, column_thresold = column_thresold, confidence_threshold = confidence_threshold)
            if len(processed) > 0:
                result.append(processed)

    return (result, confidence)

def preprocess(data, bank=None):
    result = []

    def remove_beginning_balance_duplicates(data):
        buf = []
        
        has_beginning = False
        for i in range(len(data)):
            this_page = []
            for line in data[i]:
                if any(['BEGINNING' in x.upper() for x in line]):
                    # print('FOUND BEGINNING')
                    if not has_beginning:
                        this_page.append(line)
                    has_beginning = True
                else:
                    this_page.append(line)
            buf.append(this_page)
        return buf

    def remove_ending_balance_duplicates(data):
        buf = []
        has_ending = False
        for i in reversed(range(len(data))):
            this_page = []
            for line in data[i]:
                if any(['ENDING' in x.upper() for x in line]):
                    # print('FOUND ENDING')
                    if not has_ending:
                        this_page.append(line)
                    has_ending = True
                else:
                    this_page.append(line)
            buf.insert(0, this_page)
        return buf

    if bank == None:
        for page in data:
            page = page.tolist()
            result.append(page)
    else:
        for pctr, page in enumerate(data):
            this_line = []
            this_page = []
            page = page.tolist()
            for ctr, line in enumerate(page):
                # print(f'LINE{line}')
                if (len(line) == 4 and (('BEGINNING' in (line[0]) or 'BALANCE' in (line[0])) and ('ENDING' in (line[3]) or 'BALANCE' in (line[3])))):
                    values = page[ctr+1]
                    # print(f'FOUND LINE, VALUES {values}')
                    if (pctr ==0):
                        this_line = ['BEGINNING BALANCE', values[0]]
                        this_page.append(this_line)
                    else:
                        this_line = ['ENDING BALANCE', values[3]]
                        this_page.append(this_line)
                else:
                    this_page.append(line)

            result.append(this_page)
    
    result = remove_beginning_balance_duplicates(result)
    result = remove_ending_balance_duplicates(result)
    
    return result

def ask_llm(data):
    json_str = ''
    for page in data:
        json_str += json.dumps(page)

    template = """Question: {question}

    Answer: """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], model_name='gpt-4o')
    llm_chain = prompt | llm
    question = "The data contains information for a single bank statement.  Generate a Bank Statment Summary with the output containing 'Bank Name', 'Account Number', 'Account Name', 'Address', 'Statement Period', 'Beginning Balance' using the first entry you see, 'Ending Balance' using the last entry you see, whithout doing any computation. Also include the transactions in the statement. Return the output in JSON Format, and return only the JSON text. Use the following data: "+json_str
    # question = "The data contains information for a single bank statement.  Generate a Bank Statment whithout doing any computation.  Use the following data: "+json_str
    # print(question)

    return llm_chain.invoke(question)

def _format_value(value):
    """Formats values for HTML table cells, handling nested structures."""
    if isinstance(value, list):
        return "<br>".join([_format_value(item) for item in value])
    elif isinstance(value, dict):
        return "<br>".join([f"{k}: {_format_value(v)}" for k, v in value.items()])
    else:
        return str(value)

def json_to_html_table(json_data, table_id="data_table", table_class=""):
    """
    Converts JSON data to an HTML table, handling nested lists and dictionaries.
    """

    if not json_data:
        return "<p>No data to display.</p>"

    html = f'<table id="{table_id}" class="{table_class}" >'

    keys = json_data.keys()
    print(f'KEYS {keys}')


    # Create table rows
    for key in keys:
        if (key == 'Transactions'):
            html += "<tr><td>" + key + "</td><td></td>"
            transactions = json_data[key]
            if len(transactions) > 0:
                # Assume key-values are consitent
                if type(transactions[0]) is dict:
                    keys_tr = transactions[0].keys()
                    html += "<tr>"
                    for key_tr in keys_tr:
                        html += "<td>" + key_tr+ "</td>"
                    html += "</tr>"
                    for transaction in transactions:
                        html += "<tr>"
                        for key_tr in keys_tr:
                            try:
                                html += "<td>" + transaction[key_tr] + "</td>"
                            except:
                                html += "<td></td>"
                        html += "</tr>"
                else:
                    for transaction in transactions:
                         html += "<tr><td>"
                         html += str(transaction)
                         html += "</td>></tr>"
                         



        else:
            html += "<tr><td>" + key + "</td>"
            html += f"<td>{_format_value(json_data[key])}</td>"
            html += "</tr>"
    html += "</tbody></table>"
    return html


def process_file(uploaded_file):
    if type(uploaded_file) == type(None) or len(uploaded_file) < 1:
        return ''
    pages = None

    print(f'UPLOAD {uploaded_file}')
    if is_image_pdf(uploaded_file):
        pages = convert_from_path(uploaded_file, 300)
    elif is_image_file(uploaded_file):
        pages = [Image.open(uploaded_file)]
    processed, confidence = ocr(pages, 1, row_threshold = 10, confidence_threshold=-1)
    os.remove(uploaded_file)
    # print(processed)
    confidence = confidence * 100
    pre = preprocess(processed)
    # print(pre)
    answer = ask_llm(pre)
    print(answer.content)
    print(f'CONFIDENCE = {confidence:.2f}%')
    # This function just returns the filename for now
    json_string = answer.content.replace('```','')
    if json_string.startswith('json'):
        json_string = json_string[4:]

    json_data = json.loads(json_string)
    print(json_data)
    table_data = json_to_html_table(json_data)
    print(table_data)
    return table_data
    # return f"{answer.content}\n CONFIDENCE = {confidence:.2f}%"

   # Define the interface
