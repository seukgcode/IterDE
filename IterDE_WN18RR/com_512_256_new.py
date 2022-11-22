from xml.dom.pulldom import START_DOCUMENT
import openke
from openke.config import Trainer_new, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import kd_huber_new
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the teacher model
teacher = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 512
)
teacher.load_checkpoint('./checkpoint/complEx_512.ckpt')

# define the student model
student = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 256
)

# define the loss function
model = kd_huber_new(
	model = student, 
	loss = SoftplusLoss(adv_temperature=1),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0,
    t_model = teacher
)

# train the model
trainer = Trainer_new(model = model, data_loader = train_dataloader, train_times = 500, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
student.save_checkpoint('./checkpoint/complEx_512_256_new.ckpt')

# test the model
student.load_checkpoint('./checkpoint/complEx_512_256_new.ckpt')
tester = Tester(model = student, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)