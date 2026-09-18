import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return (temperature**2) * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def distill_step(model, batch, teacher, alpha, temperature):
    input_ids, labels = batch
    device = next(model.parameters()).device
    input_ids, labels = input_ids.to(device), labels.to(device)
    output = model(input_ids)
    student_logits = output.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100).view(-1)
    flat_student = student_logits.view(-1, student_logits.size(-1))
    ce = F.cross_entropy(flat_student, shift_labels.view(-1), ignore_index=-100)
    hard_loss = ce + output.aux_loss
    if teacher is None or mask.sum() == 0:
        return hard_loss
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits[..., :-1, :].contiguous()
    flat_teacher = teacher_logits.view(-1, teacher_logits.size(-1))[:, : flat_student.size(-1)]
    soft_loss = distillation_loss(flat_student[mask], flat_teacher[mask], temperature)
    return alpha * hard_loss + (1 - alpha) * soft_loss
