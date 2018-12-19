# python plot8_mar.py --env Pusher7-v0 --iters 150 --trials 30 --arch 8 --t 100 --epochs 10 --grads 1 --weights 1 .1 .5 0 --ufact 5 --id 1 --gamma 1 --nu 0.1
# python plot7_mar10.py --env Pusher7-v0 --iters 120 --trials 10 --arch 8 --t 100 --epochs 10 --grads 1 --weights 1 .1 .5 0 --ufact 5 --id 1 --gamma 5 --nu 0.05
python plot7_mar10_reward.py --env Pusher7-v0 --iters 120 --trials 10 --arch 8 --t 100 --epochs 10 --grads 1 --weights 1 .1 .5 0 --ufact 5 --id 1 --gamma 5 --nu 0.05
#python dvrk_opt.py

