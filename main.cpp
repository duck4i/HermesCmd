#include <iostream>
#include <string>
#include <vector>
#include "llama-cpp.h"

ggml_log_level def_level = GGML_LOG_LEVEL_WARN;

void log(ggml_log_level level, const char *text)
{
    if (level >= def_level && text != nullptr)
    {
        printf("%s", text);
    }
}

void print_sys_info()
{
    std::string sys_info = llama_print_system_info();
    std::cout << "GGML backend loaded. " << sys_info << std::endl;
}

int main(int argc, char *argv[])
{
    std::string model_path = "NousResearch.Hermes-3-Llama-3.2-3B.Q4_K_M.gguf";
    std::string prompt = "How are you doing";

    std::cout << "HermesCLI - model " << model_path << " starting up." << std::endl;

    ggml_backend_load_all();
    print_sys_info();

    llama_log_set([](ggml_log_level level, const char *text, void * /*user_data*/)
                  { log(level, text); }, NULL);

    int n_predict = 32; // number of tokens to predict
    llama_model_params model_params = llama_model_default_params();

    llama_model *model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    std::cout << "Model loaded OK." << std::endl;

    const int n_prompt = -llama_tokenize(model, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(model, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
    {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    {

        llama_context_params ctx_params = llama_context_default_params();
        // n_ctx is the context size
        ctx_params.n_ctx = n_prompt + n_predict - 1;
        // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
        ctx_params.n_batch = n_prompt;
        // enable performance counters
        ctx_params.no_perf = false;

        llama_context *ctx = llama_new_context_with_model(model, ctx_params);

        if (ctx == NULL)
        {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
            return 1;
        }

        // initialize the sampler

        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler *smpl = llama_sampler_chain_init(sparams);

        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // print the prompt token-by-token

        for (auto id : prompt_tokens)
        {
            char buf[128];
            int n = llama_token_to_piece(model, id, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
        }

        // prepare a batch for the prompt

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        // main loop
        int n_decode = 0;
        llama_token new_token_id;

        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;)
        {
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }

            n_pos += batch.n_tokens;

            // sample the next token
            {
                new_token_id = llama_sampler_sample(smpl, ctx, -1);

                // is it an end of generation?
                if (llama_token_is_eog(model, new_token_id))
                {
                    break;
                }

                char buf[128];
                int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
                if (n < 0)
                {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    return 1;
                }
                std::string s(buf, n);
                printf("%s", s.c_str());
                fflush(stdout);

                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&new_token_id, 1);

                n_decode += 1;
            }
        }

        llama_sampler_free(smpl);
        llama_free(ctx);
    }

    printf("\n");
    llama_free_model(model);

    return 0;
}