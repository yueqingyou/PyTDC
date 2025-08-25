# -*- coding: utf-8 -*-

import os
import sys

import unittest
import shutil
import numpy as np

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# TODO: add verification for the generation other than simple integration

from tdc_ml.model_server.tokenizers.geneformer import GeneformerTokenizer


def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums)) + 1


class TestModelServer(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())

    def testscGPT(self):
        from tdc_ml.multi_pred.anndata_dataset import DataLoader
        from tdc_ml import tdc_hf_interface
        from tdc_ml.model_server.tokenizers.scgpt import scGPTTokenizer
        import torch
        adata = DataLoader("cellxgene_sample_small",
                           "./data",
                           dataset_names=["cellxgene_sample_small"],
                           no_convert=True).adata
        scgpt = tdc_hf_interface("scGPT")
        model = scgpt.load()  # this line can cause segmentation fault
        tokenizer = scGPTTokenizer()
        gene_ids = adata.var["feature_name"].to_numpy(
        )  # Convert to numpy array
        tokenized_data = tokenizer.tokenize_cell_vectors(
            adata.X.toarray(), gene_ids)
        mask = torch.tensor([x != 0 for x in tokenized_data[0][1]],
                            dtype=torch.bool)
        assert sum(mask) != 0, "FAILURE: mask is empty"
        first_embed = model(tokenized_data[0][0],
                            tokenized_data[0][1],
                            attention_mask=mask)
        print(f"scgpt ran successfully. here is an output {first_embed}")

    def testGeneformerPerturb(self):
        from tdc_ml.multi_pred.perturboutcome import PerturbOutcome
        dataset = "scperturb_drug_AissaBenevolenskaya2021"
        data = PerturbOutcome(dataset)
        adata = data.adata
        tokenizer = GeneformerTokenizer(max_input_size=3)
        adata.var["feature_id"] = adata.var.index.map(
            lambda x: tokenizer.gene_name_id_dict.get(x, 0))
        x = tokenizer.tokenize_cell_vectors(adata,
                                            ensembl_id="feature_id",
                                            ncounts="ncounts")
        cells, _ = x
        assert cells, "FAILURE: cells false-like. Value is = {}".format(cells)
        assert len(cells) > 0, "FAILURE: length of cells <= 0 {}".format(cells)
        from tdc_ml import tdc_hf_interface
        import torch
        geneformer = tdc_hf_interface("Geneformer")
        model = geneformer.load()
        mdim = max(len(cell) for b in cells for cell in b)
        batch = cells[0]
        for idx, cell in enumerate(batch):
            if len(cell) < mdim:
                for _ in range(mdim - len(cell)):
                    cell = np.append(cell, 0)
                batch[idx] = cell
        input_tensor = torch.tensor(batch)
        assert input_tensor.shape[0] == 512, "unexpected batch size"
        assert input_tensor.shape[1] == mdim, f"unexpected gene length {mdim}"
        attention_mask = torch.tensor([[t != 0 for t in cell] for cell in batch
                                      ])
        assert input_tensor.shape[0] == attention_mask.shape[0]
        assert input_tensor.shape[1] == attention_mask.shape[1]
        try:
            outputs = model(input_tensor,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        except Exception as e:
            raise Exception(
                f"sizes: {input_tensor.shape[0]}, {input_tensor.shape[1]}\n {e}"
            )
        num_out_in_batch = len(outputs.hidden_states[-1])
        input_batch_size = input_tensor.shape[0]
        num_gene_out_in_batch = len(outputs.hidden_states[-1][0])
        assert num_out_in_batch == input_batch_size, f"FAILURE: length doesn't match batch size {num_out_in_batch} vs {input_batch_size}"
        assert num_gene_out_in_batch == mdim, f"FAILURE: out length {num_gene_out_in_batch} doesn't match gene length {mdim}"

    def testscVI(self):
        from tdc_ml.multi_pred.anndata_dataset import DataLoader
        from tdc_ml import tdc_hf_interface

        adata = DataLoader("scvi_test_dataset",
                           "./data",
                           dataset_names=["scvi_test_dataset"],
                           no_convert=True).adata

        scvi = tdc_hf_interface("scVI")
        model = scvi.load()
        output = model(adata)
        print(f"scVI ran successfully. here is an ouput: {output}")

    def testVCGPT(self):
        from tdc_ml.model_server import load_from_hf
        vcgpt = load_from_hf("VCGPT")
        model = vcgpt.load()
        print(f"VCGPT loaded successfully")
        # data
        from tdc_ml.multi_pred.anndata_dataset import DataLoader
        from tdc_ml.utils.load import download_wrapper, pd_load
        adata = DataLoader("vcgpt_sensitivity_demo",
                           "./data",
                           dataset_names=["vcgpt_sensitivity_demo"],
                           no_convert=True).adata
        download_wrapper("vcgpt_gene_vocab", "./data", "vcgpt_gene_vocab")
        gene_vocab = pd_load("vcgpt_gene_vocab", "./data")
        # process data
        from tdc_ml.model_server.model_loaders.instructcell import unify_gene_features
        adata = unify_gene_features(
            adata,
            gene_features=gene_vocab,
            force_gene_symbol_uppercase=False,
        )
        # inference
        # Select a random single-cell sample and extract its gene counts and metadata
        k = np.random.randint(0, len(adata))
        gene_counts = adata[k, :].X.toarray()
        sc_metadata = adata[k, :].obs.iloc[0].to_dict()

        # Define the model prompt with placeholders for metadata and gene expression profile
        prompt = (
            "Given {sequencing_method}, can you predict the response of the single cell {input} from {species} when exposed to {drug} in {tissue}?"
        )

        output = model.predict(
            prompt,
            gene_counts=gene_counts,
            sc_metadata=sc_metadata,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            max_new_tokens=256,
        )
        val = output["text"]
        assert val is not None and len(val) > 0, "VCGPT output is empty"
        assert val in ["Sensitive", "Resistant",
                       "Holiday"], f"VCGPT output is unexpected: {val}"

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass
