import os
import shutil
import pandas as pd


# Colunas onde vamos trocar basic/strong
COLS_TO_FIX = ["run_id_prefix", "run_id", "model_path", "loss_history_path"]


def corrige_sumario(csv_rel_path: str, root_dir: str = None):
    """
    Corrige um arquivo de sumário:
    - Garante que 'run_id_prefix', 'run_id', 'model_path' e 'loss_history_path'
      estejam consistentes com o valor de 'entangler'.
    - Renomeia os arquivos no disco conforme os novos paths.
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(root_dir, csv_rel_path)

    print(f"\n=== Corrigindo sumário: {csv_path} ===")

    if not os.path.exists(csv_path):
        print(f"[ERRO] Arquivo CSV não encontrado: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # backup
    backup_path = csv_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copyfile(csv_path, backup_path)
        print(f"[INFO] Backup criado: {backup_path}")
    else:
        print(f"[INFO] Backup já existe: {backup_path}")

    # percorre linhas
    for idx, row in df.iterrows():
        ent = str(row["entangler"])
        prefix = str(row["run_id_prefix"])

        # já está consistente?
        if ent in prefix:
            continue

        # decide o que trocar
        wrong_tag = None
        correct_tag = None

        if "basic" in prefix and ent == "strong":
            wrong_tag = "basic"
            correct_tag = "strong"
        elif "strong" in prefix and ent == "basic":
            wrong_tag = "strong"
            correct_tag = "basic"
        else:
            print(f"[PULO] Linha {idx}: padrão desconhecido "
                  f"(run_id_prefix='{prefix}', entangler='{ent}')")
            continue

        print(f"\nCorrigindo linha {idx}: {wrong_tag} → {correct_tag}")

        # ---------- Renomear arquivos antes de alterar o DF ----------
        # Usamos exatamente o que está no CSV, apenas prefixando root_dir
        old_model_rel = str(row["model_path"])
        old_hist_rel = str(row["loss_history_path"])

        new_model_rel = old_model_rel.replace(wrong_tag, correct_tag)
        new_hist_rel = old_hist_rel.replace(wrong_tag, correct_tag)

        old_model_abs = os.path.join(root_dir, old_model_rel)
        new_model_abs = os.path.join(root_dir, new_model_rel)

        old_hist_abs = os.path.join(root_dir, old_hist_rel)
        new_hist_abs = os.path.join(root_dir, new_hist_rel)

        # modelo
        print("  model_path:")
        print(f"    {old_model_rel}")
        print(f"    -> {new_model_rel}")
        if os.path.exists(old_model_abs):
            os.rename(old_model_abs, new_model_abs)
        else:
            print(f"  [AVISO] Modelo não encontrado: {old_model_abs}")

        # histórico
        print("  loss_history_path:")
        print(f"    {old_hist_rel}")
        print(f"    -> {new_hist_rel}")
        if os.path.exists(old_hist_abs):
            os.rename(old_hist_abs, new_hist_abs)
        else:
            print(f"  [AVISO] Histórico não encontrado: {old_hist_abs}")

        # ---------- Atualizar as strings no DataFrame ----------
        for col in COLS_TO_FIX:
            old_val = str(df.at[idx, col])
            new_val = old_val.replace(wrong_tag, correct_tag)
            df.at[idx, col] = new_val

        # Só para log:
        print("  run_id_prefix / run_id atualizados para refletir o entangler.")


    # Salvar CSV corrigido
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] CSV atualizado: {csv_path}")


if __name__ == "__main__":
    # Ajuste os caminhos relativos conforme estão no seu projeto
    # (pasta 'experimentos_pinn' na raiz)
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # Para o QNN (quantico)
    corrige_sumario(os.path.join("experimentos_pinn", "sumario_quantico.csv"), ROOT)

    # Para o CQNN (cquantico) – o que você pediu agora
    corrige_sumario(os.path.join("experimentos_pinn", "sumario_cquantico.csv"), ROOT)
