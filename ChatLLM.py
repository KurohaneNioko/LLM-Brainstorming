def chat_baichuan(baichuan_model, baichuan_tokenizer, baichuan_device, prompt, history=None):
    if history is None:
         history = [{"role": "user", "content": prompt}]
    else:
        history.append({"role": "user", "content": prompt})
    try:
        response = baichuan_model.chat(baichuan_tokenizer, history)
    except Exception as e:
        print(e)
        return None, None
    history.append({"role": "assistant", "content": response})
    return response, history

def chat_qwen(qw_model, qw_tokenizer, qw_device, prompt, messages=None):
    if messages is None:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt})
    try:
        new_input = qw_tokenizer([qw_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)], return_tensors="pt").to(qw_device)
        resp = qw_tokenizer.batch_decode([output_ids[len(input_ids):] for input_ids, output_ids in zip(new_input.input_ids,   qw_model.generate(new_input.input_ids, max_new_tokens=512, pad_token_id=qw_tokenizer.eos_token_id))], 
                                     skip_special_tokens=True)[0]
    except Exception as e:
        print(e)
        return None, None
    messages.append({"role": "assistant", "content": resp})
    return resp, messages

def chat_qwen2(qw_model, qw_tokenizer, qw_device, prompt, messages=None):
    if messages is None:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt})
    
    text = qw_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)
    model_inputs = qw_tokenizer([text], return_tensors="pt").to(qw_device)
    generated_ids = qw_model.generate(
        model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = qw_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    messages.append({"role": "assistant", "content": response})
    return response, messages

def chat_phi(phi_model, phi_tokenizer, phi_device, prompt, messages=None):
    """Instruct: <prompt>\nOutput:"""
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages.append({"role": "user", "content": prompt})
    messages_str = ""
    for d in messages:
        if d['role'] == 'user':
            messages_str += f"Instruct: {d['content']}\n"
        elif d['role'] == 'assistant':
            messages_str += f"Output: {d['content']}\n"
    # messages += f"Instruct: {prompt}\nOutput:"
    inputs = phi_tokenizer(messages_str, return_tensors="pt", return_attention_mask=False).to(phi_device)
    outputs = phi_model.generate(**inputs, max_length=200)
    text = phi_tokenizer.batch_decode(outputs)[0]
    messages.append({"role": "assistant", "content": text})
    return text, messages

def chat_tinyllama(tiny_model, tiny_tokenizer, tiny_device, prompt, messages=None):
    if messages is None:
        messages = [{"role": "system", "content": "You are a helpful chatbot who can help solve problems."}]
    messages.append({"role": "user", "content": prompt})
    input_ids = tiny_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', truncation=True)
    # input_ids = input_ids[-tiny_tokenizer.model_max_length:]
    output_ids = tiny_model.generate(input_ids.to(tiny_device))
    response = tiny_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response})
    return response, messages

def chat_llama(llama_model, llama_tokenizer, llama_device, prompt, messages=None):
    if messages is None:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt})
    new_input = llama_tokenizer([llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)], return_tensors="pt").to(llama_device)
    response = llama_tokenizer.batch_decode([output_ids[len(input_ids):] for input_ids, output_ids in zip(new_input.input_ids, llama_model.generate(new_input.input_ids, max_new_tokens=512, pad_token_id=llama_tokenizer.eos_token_id))], 
                                     skip_special_tokens=True)[0]
    # input_ids = llama_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    # # input_ids = input_ids[-tiny_tokenizer.model_max_length:]
    # output_ids = llama_model.generate(input_ids.to(llama_device))
    # response = llama_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": response})
    return response, messages

def chat_mistral(mistral_model, mistral_tokenizer, mistral_device, prompt, messages=None):
    if messages is None:
         messages = [{"role": "user", "content": prompt}]
    else:
        messages.append({"role": "user", "content": prompt})
    new_input = mistral_tokenizer([mistral_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)], return_tensors="pt").to(mistral_device)
    response = mistral_tokenizer.batch_decode([output_ids[len(input_ids):] for input_ids, output_ids in zip(new_input.input_ids, mistral_model.generate(new_input.input_ids, max_new_tokens=512, do_sample=True, pad_token_id=mistral_tokenizer.eos_token_id))], 
                                 skip_special_tokens=True)[0]
    # model_inputs = mistral_tokenizer.apply_chat_template(messages, return_tensors="pt").to(mistral_device)
    # generated_ids = mistral_model.generate(model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=mistral_tokenizer.eos_token_id)
    # response = mistral_tokenizer.decode(generated_ids[0])
    messages.append({"role": "assistant", "content": response})
    return response, messages
