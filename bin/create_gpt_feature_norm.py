import os
import openai
import pandas as pd 
import argparse

def authenticate():
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")

def make_request(text, model):
    if model == 'gpt-3.5-turbo' or model == 'gpt-3.5-turbo-0301':
        messages = [{
            "role": "user",
            "content": text
        }]
        response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.5, frequency_penalty=0.33, max_tokens=70)
        response_text = response['choices'][0]['text']
    else:
        response = openai.Completion.create(engine=model, prompt=text, temperature=0.5, frequency_penalty=0.33, max_tokens=70)
        response_text = response['choices'][0]['message']['content']
    return response_text


def generate_train_sentence(train_df):
    text = ''
    train_df = train_df[:3]
    for row in train_df.itertuples():
        text += row.question
        text += ' '
        text += row.answer
    return text

def write_to_output(output_filename, answers, first_df, header=False):
    mode = 'a'
    if first_df:
        mode = 'w'
    answers.to_csv(output_filename, index=False, mode=mode, header=header)

def run(args):
    output_dir = args.output_dir
    model = args.model
    data_path = f'{output_dir}/encoded_answers_openai.csv'
    train_dir = args.train_dir 
    
    retrival_path = args.retrival_path
    retrieval_df = pd.read_csv(retrival_path)
    first_df = True 

    try:
        current_answers_saved = pd.read_csv(data_path)
        first_df = False
    except IOError:
        current_answers_saved = pd.DataFrame({'concept_id': [], 'run_nr': [], 'answer': []})
    
    for train_file_name in os.listdir(train_dir):
        print(f'Run {train_file_name}')
        run_nr = int(train_file_name.split('_')[1].split('.')[0])
        train_df = pd.read_csv('%s/%s' % (train_dir, train_file_name))
        train_text = generate_train_sentence(train_df)
            
        for row in retrieval_df.itertuples():
            if not (((current_answers_saved.concept_id == row.id) & (current_answers_saved.run_nr == run_nr)).any()): 
                question = row.question
                text = train_text + '\n%s' % question
                answer = make_request(text, model)
                answer_dict = {'concept': row.concept, 'answer': answer, 'id': row.id, 'run_nr': run_nr}
                answers_df = pd.DataFrame([answer_dict])
                answers_df = answers_df.replace('\n','', regex=True)
                write_to_output(data_path, answers_df, first_df)
                first_df = False
            break
        break

parser = argparse.ArgumentParser()
parser.set_defaults(function=run)
parser.add_argument("--output_dir", dest='output_dir')
parser.add_argument("--train_dir", dest='train_dir')
parser.add_argument("--retrival_path", dest='retrival_path')
parser.add_argument("--model", dest='model')

args = parser.parse_args()
args.function(args)