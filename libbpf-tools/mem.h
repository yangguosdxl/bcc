#ifndef __MEM_H
#define __MEM_H

#define ALLOCS_MAX_ENTRIES 1000000
#define COMBINED_ALLOCS_MAX_ENTRIES 10240
#define MAX_STACK 127

struct alloc_info {
	__u64 size;
	__u64 timestamp_ns;
	int stack_id;
	int kern_stack_size;
	int user_stack_size;
	__u64 kern_stack[MAX_STACK];
	__u64 user_stack[MAX_STACK];
};

union combined_alloc_info {
	struct {
		__u64 total_size : 40;
		__u64 number_of_allocs : 24;
	};
	__u64 bits;
};

#endif /* __MEM_H */
