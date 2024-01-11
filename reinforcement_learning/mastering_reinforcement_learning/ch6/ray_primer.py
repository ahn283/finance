import ray

# initialize ray
ray.init()

# using remote functions
@ray.remote
def remote_function():
    return 1

object_ids = []
for _ in range(4):
    y_id = remote_function()
    object_ids(y_id)
    
@ray.remote
def remote_chain_function(value):
    return value + 1

y1_id = remote_chain_function()
chained_id = remote_chain_function(y1_id)