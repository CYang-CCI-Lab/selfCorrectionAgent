from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batchinput.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)

from openai import OpenAI
client = OpenAI()

batch_input_file_id = batch_input_file.id

print(batch_input_file_id)


client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "nightly eval job"
    }
)

batch = client.batches.retrieve("batch_abc123")
print(batch)


from openai import OpenAI
client = OpenAI()

const batch = client.batches.retrieve("batch_abc123")



from openai import OpenAI
client = OpenAI()

file_response = client.files.content("file-xyz123") # output_file_id from the Batch object
print(file_response.text)