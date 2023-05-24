import os
import sys
sys.path.append('..')

import pandas as pd 
import argparse
import multiprocessing as mp 
from utils.norm_generation.claude import make_request as make_claude_request
from utils.norm_generation.openai import make_request as make_openai_request

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
    n_jobs = 3 #mp.cpu_count()
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
    for run_nr in list(range(1,31)):  
        run_nr = str(run_nr)
        train_file_name = f"train_{run_nr}.csv"
        print(f'Check {train_file_name}')
        train_df = pd.read_csv('%s/%s' % (train_dir, train_file_name))

        for row in retrieval_df.itertuples():
            concept_id = row.id
            concept_run_already_sampled = (((current_answers_saved.concept_id == concept_id) & (current_answers_saved.run_nr == run_nr)).any())
            concept_occurs_in_priming = concept_id in list(train_df[:3]['concept'])

            if concept_run_already_sampled or concept_occurs_in_priming:
                print(f'Skip {concept_id} - {run_nr}')
                continue
    
            question = row.question
            
            priming = []
            for priming_example in train_df.itertuples():
                priming.append([priming_example.question, priming_example.answer])

            job = {
                        'priming': priming,
                        'run_nr': run_nr,
                        'concept': row.concept,
                        'concept_id': concept_id,
                        'question': question
            }
            job_queue.put(job)
            

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
    print(f'start worker {i}')
    while 1:
        job = job_queue.get()
        if job == 'kill':
            break

        priming = job['priming']
        question = job['question']
        answer = make_claude_request(priming, model, question)
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