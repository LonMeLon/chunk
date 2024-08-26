import os

from argparse import ArgumentParser
import utils
import time



def main(args):
    # func
    ### dataset
    dataloader_func = utils.dataloader(args)
    ### retriever
    retriever_func = utils.retriever(args)
    ### reranker
    reranker_func = utils.reranker(args)
    ### generator 
    generator_func = utils.generator(args)
    ### query-chunk aggregator
    aggregator_func = utils.query_chunk_aggregator(args)

    
    # dataset
    qa_dataset, retrieve_dataset = dataloader_func.load_dataset()

    train_qa_dataset = qa_dataset['train']
    val_qa_dataset = qa_dataset['validation']

    # index retrieve_dataset
    retriever_func.index(retrieve_dataset)


    # 


    
    

    '''
    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    print(train_dataset)
    print(val_dataset)

    for item in val_dataset:
        print(item)
    '''

    
    ##### chunks
    '''

    # retrieve

    # rerank

    # chunk aggregator
    candidate_doc_set = zip(doc_id, doc_text)



    # generate
    timestep = time.time()
    questions = ['' + qqq for qqq in test_query_text]
    answers = generator_func.generate(questions)
    print('-----', 'generator time cost:', time.time() - timestep, 's')
    '''






if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()

    # dataset
    ### dataset
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default="/home/meilang/1_project/chunk/dataset/nq_open", 
                        help='[nq_open, hotpotqa]',
    )
    ### chunks
    
    
    # retriever
    parser.add_argument('--retriever', type=str, default="/home/meilang/1_project/chunk/dataset/bge-large-en-v1.5")
    ## retriever_dataset
    parser.add_argument('--retrieve_dataset', type=str, default="/home/meilang/1_project/chunk/dataset/enwiki-dec2021/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl")
    ### index
    parser.add_argument('--retriever_index_path', type=str, default="/home/meilang/1_project/chunk/dataset/enwiki-dec2021/corpora/wiki/enwiki-dec2021" )
    ### top-k
    parser.add_argument('--retriever_top_k', type=int, default=20)
    ### batch inference
    parser.add_argument('--retriever_batch_size', type=int, default=64)
    
    
    
    # reranker
    parser.add_argument('--reranker_path', type=str, default="")

    # generator
    parser.add_argument('--generator_path', type=str, default='/root/models/Qwen2-7B-Instruct')
    parser.add_argument('--generator_url', type=str, default='')



    # query-chunk aggregator
    parser.add_argument('--query_chunk_aggregator_path', type=str, default='')

    
    args = parser.parse_args()
    print(args)

    
    # main
    main(args)