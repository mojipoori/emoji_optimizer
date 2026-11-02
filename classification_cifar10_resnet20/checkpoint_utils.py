import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, beta3=0.9999, gamma=1e-3, eps=1e-8, weight_decay=5e-4,
                  reset=False, run=0, weight_decouple=False, rectify=False, fixed_decay=False, amsgrad=False, amsbound=False):
    """
    تولید نام چک‌پوینت بر اساس پارامترهای مدل و بهینه‌ساز
    """
    name = {
        'sgd': 'lr{}-momentum{}-wdecay{}-run{}'.format(lr, momentum, weight_decay, run),
        'sgdw': 'lr{}-mom{}-wd{}-run{}'.format(lr, momentum, weight_decay, run),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'fromage': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'radam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        'adabelief': 'lr{}-betas{}-{}-eps{}-wdecay{}-rectify{}-fixed_decay{}-amsgrad{}-run{}'.format(
            lr, beta1, beta2, eps, weight_decay, rectify, fixed_decay, amsgrad, run),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-amsbound{}-run{}'.format(
            lr, beta1, beta2, final_lr, gamma, weight_decay, False, run),
        'adaboundw': 'lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-eps{}-amsbound{}-run{}'.format(
            lr, beta1, beta2, final_lr, gamma, weight_decay, eps, amsbound, run),
        'yogi': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'msvag': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'moji': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'mojib': 'lr{}-betas{}-{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, beta3, eps, weight_decay, run),
        'mojic': 'lr{}-betas{}-{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, beta3, eps, weight_decay, run),
    }[optimizer]
    return '{}-{}-{}-reset{}'.format(model, optimizer, name, str(reset))


def load_checkpoint(ckpt_name):
    """
    بارگذاری چک‌پوینت از فایل
    """
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    if not os.path.exists(path):
        print('Error: checkpoint {} not found'.format(path))
        return None
    return torch.load(path)


def save_checkpoint(state, ckpt_name):
    """
    ذخیره چک‌پوینت در فایل
    """
    print('Saving checkpoint for epoch {}..'.format(state['epoch']))
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, os.path.join('checkpoint', ckpt_name))


def save_curve_data(train_accuracies, test_accuracies, ckpt_name):
    """
    ذخیره داده‌های منحنی (accuracy) در فایل
    """
    if not os.path.isdir('curve'):
        os.mkdir('curve')
    torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
               os.path.join('curve', ckpt_name))


def load_curve_data(ckpt_name):
    """
    بارگذاری داده‌های منحنی از فایل
    """
    curve = os.path.join('curve', ckpt_name)
    if os.path.exists(curve):
        curve_data = torch.load(curve)
        return curve_data['train_acc'], curve_data['test_acc']
    else:
        return [], []


def check_config_compatibility(args, ckpt):
    """
    بررسی سازگاری تنظیمات چک‌پوینت با تنظیمات فعلی
    """
    if not ckpt:
        return False
    ckpt_args = ckpt.get('args', {})
    ckpt_params = ['model', 'optim', 'lr', 'final_lr', 'momentum', 'beta1', 'beta2', 'beta3', 'gamma', 'eps', 'weight_decay', 'reset', 'run']
    other_params = ['batchsize']
    
    for param in ckpt_params + other_params:
        if ckpt_args.get(param) != vars(args).get(param):
            print(f"Incompatible parameter: {param}, Checkpoint: {ckpt_args.get(param)}, Current: {vars(args).get(param)}")
            return False
    return True


def create_tensorboard_writer(ckpt_name):
    """
    ایجاد writer برای تنسوربورد
    """
    log_dir = os.path.join('runs', ckpt_name)
    return SummaryWriter(log_dir=log_dir)


def log_training_metrics(writer, train_loss, train_acc, test_loss, test_acc, epoch):
    """
    ثبت متریک‌های آموزش و تست در تنسوربورد
    """
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)


def initialize_training_state(args, ckpt_name):
    """
    مقداردهی اولیه وضعیت آموزش
    """
    ckpt = load_checkpoint(ckpt_name)
    writer = create_tensorboard_writer(ckpt_name)
    
    if args.resume and ckpt and check_config_compatibility(args, ckpt):
        print('Resuming from compatible checkpoint')
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
        train_accuracies, test_accuracies = load_curve_data(ckpt_name)
    else:
        if args.resume and ckpt:
            print('Checkpoint found but settings are incompatible. Starting new run.')
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_accuracies = []
        test_accuracies = []
    
    return ckpt, writer, best_acc, start_epoch, train_accuracies, test_accuracies


def update_training_state(state, net, test_acc, epoch, train_accuracies, test_accuracies, ckpt_name):
    """
    به‌روزرسانی وضعیت آموزش و ذخیره چک‌پوینت
    """
    # به‌روزرسانی state
    state.update({
        'net': net.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
    })
    
    # ذخیره چک‌پوینت
    save_checkpoint(state, ckpt_name)
    
    # به‌روزرسانی و ذخیره منحنی‌ها
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    save_curve_data(train_accuracies, test_accuracies, ckpt_name)
    
    return train_accuracies, test_accuracies 