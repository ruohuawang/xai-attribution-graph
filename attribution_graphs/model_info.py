from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('C:\\codes\\llms\\qwen05b', trust_remote_code=True)
print('Model type:', type(model).__name__)
print('\nModel attributes:')
for attr in dir(model):
    if not attr.startswith('_') and not callable(getattr(model, attr)):
        print(attr)

# 打印模型的详细结构
print('\nModel structure:')
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"- {attr}")

# 检查模型是否有model属性
if hasattr(model, 'model'):
    print('\nModel has model attribute')
    print('Model.model attributes:')
    for attr in dir(model.model):
        if not attr.startswith('_') and not callable(getattr(model.model, attr)):
            print(attr)

# 尝试找出模型的层结构
print('\nSearching for layers structure...')
if hasattr(model, 'model'):
    if hasattr(model.model, 'layers'):
        print('Found: model.model.layers')
    elif hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
        print('Found: model.model.encoder.layers')
    elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        print('Found: model.model.decoder.layers')
elif hasattr(model, 'transformer'):
    if hasattr(model.transformer, 'layers'):
        print('Found: model.transformer.layers')
    elif hasattr(model.transformer, 'h'):
        print('Found: model.transformer.h')
elif hasattr(model, 'encoder'):
    if hasattr(model.encoder, 'layers'):
        print('Found: model.encoder.layers')
elif hasattr(model, 'decoder'):
    if hasattr(model.decoder, 'layers'):
        print('Found: model.decoder.layers')