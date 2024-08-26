import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys

import time


from transformers import AutoTokenizer, AutoModel
import torch
import faiss

import datasets


from tqdm import tqdm
import gzip
import json
import csv
import requests
import numpy as np





class dataloader:
    def __init__(self, args,):
        self.args = args

    def load_dataset(self, ):
        qa_dataset = None
        retrieve_dataset = None

        # qa_dataset
        timestep = time.time()
        if 'nq_open' in self.args.dataset_path:
            qa_dataset = datasets.load_dataset(self.args.dataset_path)
        print('##### load qa_dataset:', )
        print('      cost time:', time.time() - timestep, )
        print('      info:', qa_dataset, )

        # retrieve_dataset
        timestep = time.time()
        if 'enwiki-dec2021' in self.args.retrieve_dataset:
            retrieve_dataset = []
            # 33176581 records
            with open(self.args.retrieve_dataset, 'r') as file:
                for i, line in tqdm(enumerate(file)):
                    json_content = json.loads(line)
                    retrieve_dataset.append(
                        json_content['text'],
                    )
                    if i > 5000:
                        break
        print('##### load retrieve_dataset:', )
        print('      cost time:', time.time() - timestep, )
        print('      info length:', len(retrieve_dataset), )

        return qa_dataset, retrieve_dataset





class retriever:
    def __init__(self, args,):
        self.args = args
        
        if 'bge-large-en-v1.5' in self.args.retriever:
            self.device = torch.device("cuda", 0)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.retriever)
            self.retriever_model = AutoModel.from_pretrained(self.args.retriever).to(self.device)
            self.retriever_model.eval()


    def encode_query(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
        model_output = self.retriever_model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def encode_doc(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
        model_output = self.retriever_model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def index(self, texts):
        # inference text doc
        timestep = time.time()
        texts_embs = []
        data_loader = torch.utils.data.DataLoader(
            texts,
            batch_size = self.args.retriever_batch_size,
            collate_fn = lambda x:x,
            shuffle=False,
        )
        with torch.no_grad():
            for iii, batch in tqdm(enumerate(data_loader)):
                batch_texts_embs = self.encode_doc(batch)
                texts_embs.append(batch_texts_embs.cpu().numpy())
        texts_embs = np.concatenate(texts_embs, 0)
        #texts_embs = np.concatenate(texts_embs, 0).astype('float32')
        print('##### inference text doc:', )
        print('      cost time:', time.time() - timestep, )
        print('      info length:', texts_embs.shape, )


        if not os.path.exists(self.args.retriever_index_path):
            os.makedirs(self.args.retriever_index_path)

        timestep = time.time()
        faiss_index = faiss.IndexFlatIP(texts_embs.shape[-1])
        faiss_index.add(texts_embs)
        faiss.write_index(faiss_index, self.args.retriever_index_path + '/IndexFlatIP.bin')
        print('##### write faiss index:', )
        print('      cost time:', time.time() - timestep, )
        
        '''
        # build index
        while True:
            rebuild = input('re-build chunk index? (y or n)')
            if rebuild in ('y', 'n'):
                break
            else:
                print('wrong choice!')
        
        if rebuild == 'y':
            timestep = time.time()
            faiss_index = faiss.IndexFlatIP(texts_embs.shape[-1])
            faiss_index.add(texts_embs)
            faiss.write_index(faiss_index, self.args.retriever_index_path + '/IndexFlatIP.bin')
            print('##### write faiss index:', )
            print('      cost time:', time.time() - timestep, )
        '''

    def retrieve(self, queries, text_docs):
        # inference queries
        queries_embs = []
        data_loader = torch.utils.data.DataLoader(
            queries,
            batch_size = self.args.retriever_batch_size,
            collate_fn = lambda x:x,
            shuffle=False,
        )
        with torch.no_grad():
            for iii, batch in tqdm(enumerate(data_loader)):
                batch_queries_embs = self.encode_query(batch)
                queries_embs.append(batch_queries_embs.cpu().numpy())
        queries_embs = np.concatenate(queries_embs, 0).astype('float32')

        # process search
        faiss_index = faiss.read_index(self.args.retriever_index_path + '/IndexFlatIP.bin')
        singlegpu = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(singlegpu, 0, faiss_index)
        top_docs_score, top_docs_position = faiss_index.search(test_query_emb, self.args.retrieve_top_k)

        # get docs
        top_docs = []
        for i in range(top_docs_position.shape[0]):
            i_top_docs = [ text_docs[posit] for posit in top_docs_position[i] ]
            top_docs.append(i_top_docs)

        return top_docs
        


class reranker:
    def __init__(self, args,):
        self.args = args

    def rerank(self, query, docs):
        pass


class generator:
    def __init__(self, args,):
        self.args = args

    def prompt(self, context, question):
        pass

    def generate(self, questions,):
        # request
        headers = {'Content-Type': 'application/json'}
        data = {
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": qqq,
                } 
                for qqq in questions
            ]
        }

        # response
        response = requests.post(
            self.args.url, 
            headers=headers, 
            json=data, 
        )

        return response


class query_chunk_aggregator:
    def __init__(self, args,):
        self.args = args

    
