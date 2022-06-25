from opensearchpy import OpenSearch, RequestsHttpConnection


host = 'search-face-and-action-y6b5mzj5mupvipkof4pdaqlidy.us-east-2.es.amazonaws.com' # cluster endpoint, for example: my-test-domain.us-east-1.es.amazonaws.com
region = 'us-east-2'
port = 443

auth = ('master', '@Master1234')

# Create the client with SSL/TLS enabled, but hostname verification disabled.
client = OpenSearch(
    hosts = [{'host': host,'port':port}],
#     http_compress = True, # enables gzip compression for request bodies
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

def check_client():
    print("Checking client status")
    return client.info()


def create_or_update(index,id,body):
    if client.exists(index,id):
        print("Doc Id [%s] already exists in index [%s]" % (id,index))
#         print("Deleting previous one")
        client.delete(index,id)
#         client.update(index,id,body)
    else:
        print("Creating new document with id %s for index %s" % (id,index))
    
    client.create(index=index,body=body,id=id)

def query_by_name(index,name):
    raise NotImplemented

# def create_
