import openke
from openke.config import Trainer_new, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import kd_huber_new
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0,
	batch_size = 1024)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the student model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 256, 
	p_norm = 1, 
	norm_flag = True)

# define the teacher model
t_transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 512,
	p_norm = 1,
	norm_flag = True
)

t_transe.load_checkpoint('./checkpoint/transe_512.ckpt')

# define the loss function
model = kd_huber_new(
	model = transe,
	t_model = t_transe,
	loss = MarginLoss(margin = 5.0, adv_temperature=1),
	batch_size = train_dataloader.get_batch_size(),
	l = 0.001
)

# h:s 1:10

# train the model
trainer = Trainer_new(model = model, data_loader = train_dataloader, train_times = 500, alpha = 0.5, use_gpu = True, opt_method = 'Adagrad')
trainer.run()
transe.save_checkpoint('./checkpoint/transe_512_256_new.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe_512_256_new.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

