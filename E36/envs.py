from environment import envs

class Regression(envs.OneStep):
	def pos_sample(self):
		self.a = self.true_f-self.noise_f
		self.f = self.true_f
		self.fake = self.ae.decode(self.f)
		self.fake_patch = self.fake.masked_select(self.patch_mask).view(self.bs, 3, self.dx[0], self.dy[0])
		return self.state,self.a,self.reward,self.done
	def step(self,a):
		self.f=  self.noise_f+a
		self.fake = self.ae.decode(self.f)
		self.fake_patch = self.fake.masked_select(self.patch_mask).view(self.bs, 3, self.dx[0], self.dy[0])
		return self.state, self.reward, self.done

class E35(envs.OneStep):
	@property
	def r_l1(self):
		return -(self.true_f-self.f).view(self.bs, -1).abs().mean(dim=1,keepdim=True)

	def pos_sample(self):
		self.a = self.true_f-self.noise_f
		self.f = self.true_f
		self.fake = self.ae.decode(self.f)
		self.fake_patch = self.fake.masked_select(self.patch_mask).view(self.bs, 3, self.dx[0], self.dy[0])
		return self.state,self.a,self.reward,self.done