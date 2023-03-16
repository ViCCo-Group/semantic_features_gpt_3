import os
import openai
import pandas as pd 
import argparse
import multiprocessing as mp 

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
        response_text = response['choices'][0]['message']['content']
    else:
        pass
        #response = openai.Completion.create(engine=model, prompt=text, temperature=0.5, frequency_penalty=0.33, max_tokens=70)
        #response_text = response['choices'][0]['text']
    return response_text


def generate_train_sentence(train_df):
    text = ''
    train_df = train_df[:3]
    for row in train_df.itertuples():
        text += row.question
        text += ' '
        text += row.answer
    return text


def run(args):
    output_dir = args.output_dir
    model = args.model
    data_path = f'{output_dir}/encoded_answers_openai.csv'
    train_dir = args.train_dir 
    retrival_path = args.retrival_path
    retrieval_df = pd.read_csv(retrival_path)

    try:
        current_answers_saved = pd.read_csv(data_path, names=['concept', 'answer', 'concept_id', 'run_nr'])
    except IOError:
        current_answers_saved = pd.DataFrame({'concept_id': [], 'run_nr': [], 'answer': []})

    print(current_answers_saved)
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    output_queue = manager.Queue()    
    job_queue  = manager.Queue()  
    n_jobs = 15 #mp.cpu_count()  
    pool = mp.Pool(n_jobs)

    #put ouput listener to work first
    print('Start Output Writer')
    watcher = pool.apply_async(listener, (output_queue, data_path))

    #fire off workers
    print('Start Workers')
    jobs = []
    for i in range(n_jobs):
        job = pool.apply_async(worker, (i, job_queue, output_queue, model))
        jobs.append(job)

    # create jobs
    for train_file_name in os.listdir(train_dir):
        print(f'Check {train_file_name}')
        run_nr = int(train_file_name.split('_')[1].split('.')[0])
        train_df = pd.read_csv('%s/%s' % (train_dir, train_file_name))
        train_text = generate_train_sentence(train_df)
        for row in retrieval_df.itertuples():
            if not (((current_answers_saved.concept_id == row.id) & (current_answers_saved.run_nr == run_nr)).any()): 
                question = row.question
                text = train_text + '\n%s' % question
                job = {
                        'text': text,
                        'run_nr': run_nr,
                        'concept': row.concept,
                        'concept_id': row.id
                }
                job_queue.put(job)
            else:
                print(f'Skip {row.concept} - {run_nr}')

    for _ in jobs:
        job_queue.put('kill')

    print(f'Number of jobs: {job_queue.qsize()}')
        
    print('Wait for workers to finish')
    for job in jobs:
        job.get()

    print('Workers are done!')
    output_queue.put('kill')
    watcher.get()
    print('Output Writer is done!')

    pool.close()
    pool.join()



def escape_answer(text):
    return text.replace('\n', '').replace('"', '""')

def worker(i, job_queue, output_queue, model):
    while 1:
        job = job_queue.get()
        if job == 'kill':
            break

        text = job['text']
        answer = make_request(text, model)
        answer = escape_answer(answer)
        job['answer'] = answer
        output_queue.put(job)
    return True 

def listener(output_queue, out_path):
    '''listens for messages on the q, writes to file. '''
    with open(out_path, 'a+') as f:
        while 1:
            result = output_queue.get()
            if result == 'kill':
                break

            cocnept = result['concept']
            answer = result['answer']
            cocnept_id = result['concept_id']
            run_nr = result['run_nr']
            text = f'{cocnept},"{answer}",{cocnept_id},{run_nr}'
            f.write(text + '\n')
            f.flush()

    
    return True

    

parser = argparse.ArgumentParser()
parser.set_defaults(function=run)
parser.add_argument("--output_dir", dest='output_dir')
parser.add_argument("--train_dir", dest='train_dir')
parser.add_argument("--retrival_path", dest='retrival_path')
parser.add_argument("--model", dest='model')

args = parser.parse_args()
args.function(args)