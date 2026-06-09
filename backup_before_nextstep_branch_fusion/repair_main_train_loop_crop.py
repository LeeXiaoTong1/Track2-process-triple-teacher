from pathlib import Path
import re
import shutil

p = Path("main_train.py")
bak = Path("main_train.py.bak_before_loop_repair")
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"[backup] {p} -> {bak}")

s = p.read_text(encoding="utf-8")

if "def forward_maybe_multicrop" not in s:
    raise RuntimeError("forward_maybe_multicrop() is missing. Run the robust crop patch first.")

new_loop = r'''    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)

        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                batch = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                batch = next(trainOri_flow)

            feat, audio_fn, labels, type_ids = unpack_batch(batch, args.device)

            # ==========================================================
            # SAM / ASAM / CSAM branch: keep original behavior.
            # Do not use multi-crop training together with SAM branches.
            # ==========================================================
            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)

                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()

                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)

                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()

                feat_optimizer.second_step(zero_grad=True)

                trainlossDict["base_loss"].append(feat_loss.item())

                with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                    log.write(
                        str(epoch_num)
                        + "\t"
                        + str(i)
                        + "\t"
                        + str(trainlossDict[monitor_loss][-1])
                        + "\n"
                    )

                continue

            # ==========================================================
            # Normal branch: supports [B, L] and [B, K, L] multi-crop.
            # ==========================================================
            feat_optimizer.zero_grad()

            feats, feat_outputs, crop_cons_loss = forward_maybe_multicrop(
                args,
                feat_model,
                feat
            )

            # Detection loss
            if (
                args.train_task == "atadd-track2"
                and args.t2_gdro
                and type_ids is not None
            ):
                feat_loss = type_group_dro_ce_loss(
                    outputs=feat_outputs,
                    labels=labels,
                    type_ids=type_ids,
                    class_weight=weight,
                    eta=args.t2_gdro_eta,
                    n_types=4
                )
            else:
                feat_loss = criterion(feat_outputs, labels)

            # Type-adversarial loss
            if (
                args.train_task == "atadd-track2"
                and args.t2_type_adv
                and type_ids is not None
                and hasattr(feat_model, "type_adv_head")
            ):
                z = pool_for_type_head(feats)
                type_logits_adv = feat_model.type_adv_head(
                    grad_reverse(z, args.t2_grl_lambda)
                )
                loss_type_adv = F.cross_entropy(type_logits_adv, type_ids)
                feat_loss = feat_loss + args.t2_type_adv_weight * loss_type_adv

            # Router type supervision
            if (
                args.train_task == "atadd-track2"
                and args.t2_router_type_loss > 0
                and type_ids is not None
                and hasattr(feat_model, "latest_type_logits")
                and feat_model.latest_type_logits is not None
            ):
                loss_router_type = F.cross_entropy(
                    feat_model.latest_type_logits,
                    type_ids
                )
                feat_loss = feat_loss + args.t2_router_type_loss * loss_router_type

            # Router entropy
            if (
                args.train_task == "atadd-track2"
                and args.t2_router_entropy > 0
                and hasattr(feat_model, "latest_expert_weights")
                and feat_model.latest_expert_weights is not None
            ):
                ent = router_entropy_loss(feat_model.latest_expert_weights)
                feat_loss = feat_loss - args.t2_router_entropy * ent

            # UFM auxiliary type loss
            if (
                args.train_task == "atadd-track2"
                and getattr(args, "ufm_type_loss", 0) > 0
                and type_ids is not None
                and hasattr(feat_model, "latest_type_logits")
                and feat_model.latest_type_logits is not None
            ):
                loss_ufm_type = F.cross_entropy(
                    feat_model.latest_type_logits,
                    type_ids
                )
                feat_loss = feat_loss + args.ufm_type_loss * loss_ufm_type

            # UFM router entropy
            if (
                args.train_task == "atadd-track2"
                and getattr(args, "ufm_router_entropy", 0) > 0
                and hasattr(feat_model, "latest_expert_weights")
                and feat_model.latest_expert_weights is not None
            ):
                ent = router_entropy_loss(feat_model.latest_expert_weights)
                feat_loss = feat_loss - args.ufm_router_entropy * ent

            # Multi-crop consistency loss
            if getattr(args, "crop_consistency_weight", 0.0) > 0:
                feat_loss = feat_loss + args.crop_consistency_weight * crop_cons_loss

            if not torch.isfinite(feat_loss):
                print(
                    f"[skip non-finite loss] epoch={epoch_num}, step={i}, "
                    f"loss={feat_loss.item()}"
                )
                feat_optimizer.zero_grad(set_to_none=True)
                continue

            feat_loss.backward()

            bad_grads = find_bad_grads(feat_model)

            if len(bad_grads) > 0:
                print(f"[STOP] non-finite gradients at epoch={epoch_num}, step={i}")
                feat_optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(f"Non-finite gradients found: {bad_grads[:10]}")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                feat_model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=False
            )

            if not torch.isfinite(grad_norm):
                print(
                    f"[skip non-finite grad_norm] epoch={epoch_num}, step={i}, "
                    f"grad_norm={grad_norm}"
                )
                feat_optimizer.zero_grad(set_to_none=True)
                continue

            feat_optimizer.step()

            bad_params = find_bad_params(feat_model)

            if len(bad_params) > 0:
                print(f"[STOP] non-finite parameters after step epoch={epoch_num}, step={i}")
                raise RuntimeError(f"Non-finite parameters found: {bad_params[:10]}")

            trainlossDict["base_loss"].append(feat_loss.item())

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(
                    str(epoch_num)
                    + "\t"
                    + str(i)
                    + "\t"
                    + str(trainlossDict[monitor_loss][-1])
                    + "\n"
                )

        # ==============================================================
        # Validation
        # ==============================================================
        feat_model.eval()

        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader, pred_loader, type_loader = [], [], [], [], [], []

            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    batch = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    batch = next(valOri_flow)

                feat, audio_fn, labels, type_ids = unpack_batch(batch, args.device)

                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = torch.sigmoid(feat_outputs[:, 0])
                    pred = torch.where(
                        score >= 0.5,
                        torch.zeros_like(labels),
                        torch.ones_like(labels)
                    )
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    prob = F.softmax(feat_outputs, dim=1)
                    score = prob[:, 0]
                    pred = torch.where(
                        score >= 0.5,
                        torch.zeros_like(labels),
                        torch.ones_like(labels)
                    )

                ip1_loader.append(feats)
                idx_loader.append(labels)
                pred_loader.append(pred)
                devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)

                if type_ids is not None:
                    type_loader.append(type_ids)

            desc_str = ""
            for key in sorted(devlossDict.keys()):
                desc_str += key + ":%.5f" % (np.nanmean(devlossDict[key])) + ", "

            valLoss = np.nanmean(devlossDict[monitor_loss])

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels_np = torch.cat(idx_loader, 0).data.cpu().numpy()
            preds_np = torch.cat(pred_loader, 0).data.cpu().numpy()

            val_eer = em.compute_eer(
                scores[labels_np == 0],
                scores[labels_np == 1]
            )[0]

            if args.train_task == "atadd-track2" and len(type_loader) > 0:
                type_ids_np = torch.cat(type_loader, 0).data.cpu().numpy()
                val_f1, type_f1s = track2_macro_f1_by_type(
                    labels=labels_np,
                    preds=preds_np,
                    type_ids=type_ids_np,
                    n_types=4
                )
                print("Track2 Type F1s [speech, sound, singing, music]:", type_f1s)
            else:
                val_f1 = f1_score(
                    labels_np,
                    preds_np,
                    average="macro",
                    zero_division=0
                )

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(
                    str(epoch_num)
                    + "\t"
                    + str(valLoss)
                    + "\t"
                    + str(val_eer)
                    + "\t"
                    + str(val_f1)
                    + "\n"
                )

            print("Val Loss: {}".format(valLoss))
            print("Val EER: {}".format(val_eer))
            print("Val F1 : {}".format(val_f1))

        if (epoch_num + 1) % 5 == 0:
            torch.save(
                feat_model.state_dict(),
                os.path.join(
                    args.out_fold,
                    "checkpoint",
                    "atadd_model_%d.pt" % (epoch_num + 1)
                )
            )

        save_flag = False

        if args.save_best_by == "loss":
            if valLoss < prev_loss:
                prev_loss = valLoss
                save_flag = True
        elif args.save_best_by == "eer":
            if val_eer < prev_eer:
                prev_eer = val_eer
                save_flag = True
        elif args.save_best_by == "f1":
            if val_f1 > prev_f1:
                prev_f1 = val_f1
                save_flag = True

        if save_flag:
            torch.save(
                feat_model.state_dict(),
                os.path.join(args.out_fold, "atadd_model.pt")
            )
            print(f"Best model updated by {args.save_best_by} at epoch {epoch_num}")

    return feat_model'''

pattern = r'(?ms)^    for epoch_num in tqdm\(range\(args\.num_epochs\)\):.*?^    return feat_model'
s2, n = re.subn(pattern, new_loop, s, count=1)

if n != 1:
    raise RuntimeError(f"Failed to replace train loop. matched={n}")

p.write_text(s2, encoding="utf-8")
print("[done] repaired full train/validation loop")
