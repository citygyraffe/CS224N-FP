import random
from itertools import cycle

# SETTINGS
DEBUG_OUTPUT = False

class TaskScheduler:
    '''
    TaskScheduler is a class that schedules tasks to be run on the data based on the policy.

    Policies: random, round-robin, annealed-sampling

    '''
    def __init__(self, args, tasks, task_lengths):
        self.epoch_counter = 0
        self.task_counter = {task:0 for task in tasks}
        self.tasks = tasks
        self.next_task = None
        self.args = args
        
        # Used for annealed sampling
        self.num_epochs = args.epochs
        self.task_lengths = task_lengths
        self.alpha = 1
        self.task_probabilities = []
        self.total_task_length = sum(task_lengths)
        self.annealed_sampling_tasks = []
        self.annealed_sampling_cycle = None

        print(f"TaskScheduler initialized with tasks: {tasks} with lengths: {task_lengths}")

        assert self.args.scheduling_policy in ['random', 'round-robin', 'annealed-sampling'], f"Policy {self.args.policy} not supported."

    def annealed_sampling(self):
        return next(self.annealed_sampling_cycle)
    
    def step_epoch(self):
        self.alpha = 1 - 0.8 * (self.epoch_counter / self.num_epochs)
        self.task_probabilities = []
        for task_length in self.task_lengths:
            '''
            In sampling mode epoch, we want to sample tasks with a probability inversely proportional to the task length so that we can sample smaller tasks more often.
            We want to run through the smaller datasets more times so that we dont get lost in the influence of the larger datasets.

            In sampling mode batch, we want to sample tasks with a probability proportional to the task length so that more data is processed for longer tasks.
            We dont want to run through the smaller datasets many more times that the larger datasets as that could result in overfitting.
            '''
            if(self.args.scheduling_mode == 'epoch'):
                self.task_probabilities.append(1/((task_length/self.total_task_length) ** self.alpha))
            elif(self.args.scheduling_mode == 'batch'):
                self.task_probabilities.append((task_length) ** self.alpha)
            else:
                raise ValueError(f"Mode {self.args.scheduling_mode} not supported.")
            
        sample_count = self.args.num_batches_per_epoch if self.args.scheduling_mode == 'batch' else 1
        self.annealed_sampling_tasks = random.choices(self.tasks, weights=self.task_probabilities, k=sample_count)
        self.annealed_sampling_cycle = cycle(self.annealed_sampling_tasks)

        if DEBUG_OUTPUT:
            print(f"Task probs: {self.task_probabilities}")
            print(f"Annealed sampling alpha: {self.alpha}")
            print(f"Annealed sampling tasks: {self.annealed_sampling_tasks}")
        
        self.epoch_counter += 1
        
    def print_task_distribution(self):
        print(f"TaskScheduler task distribution: {self.task_counter}")

    def get_next_task(self):
        if self.args.force_task != '':
            self.next_task = self.args.force_task
        elif self.args.scheduling_policy == 'round-robin':
            self.next_task = self.tasks[self.epoch_counter % len(self.tasks)]
        elif self.args.scheduling_policy == 'annealed-sampling':
            self.next_task = self.annealed_sampling()
        elif self.args.scheduling_policy == 'random':
            self.next_task = random.choice(self.tasks)
        else:
            raise ValueError(f"Policy {self.args.scheduling_policy} not supported.")
        
        if(self.args.scheduling_mode == 'epoch'):
            print(f"Epoch {self.epoch_counter}: Next task is {self.next_task}, scheduling mode: {self.args.scheduling_policy}, force task: {self.args.force_task}")
    
        self.task_counter[self.next_task] += 1
        return self.next_task

    def __del__(self):
        self.print_task_distribution()