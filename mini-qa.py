import os
from dotenv import load_dotenv
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

load_dotenv()

ASTRA_DB_CLIENT_ID = os.getenv('ASTRA_DB_CLIENT_ID')
ASTRA_DB_CLIENT_SECRET = os.getenv('ASTRA_DB_CLIENT_SECRET')
ASTRA_DB_KEYSPACE = os.getenv('ASTRA_DB_KEYSPACE')
ASTRA_DB_SECURE_BUNDLE_PATH = os.getenv('ASTRA_DB_SECURE_BUNDLE_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

cloud_config = {
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(
    ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
myEmbeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

myCassandraVSStore = Cassandra(
    embedding=myEmbeddings,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name='embedding_tutorial'
)

# comment this out if you already have the table created
print('Loading dataset...')
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset['text'][:50]
print('Done loading dataset.')
myCassandraVSStore.add_texts(headlines)


vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVSStore)

first_question = True
while True:
    if first_question:
        query_text = input("Enter your question (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("Enter your question (or type 'quit' to exit): ")

    if query_text == 'quit':
        break

    print("QUESTION: \"%s\"" % query_text)
    answer = vectorIndex.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in myCassandraVSStore.similarity_search_with_score(query_text, k=4):
        print("DOC: \"%s\" SCORE: %f" % (doc, score))
