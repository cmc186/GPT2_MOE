def moe_mask(layers:int, schedule:str, offset:int = 1)->list[bool]:
    """  
        Function aimed to point out 
        where MOE blocks should be 
        included. 

        The function takes the total 
        amount of layers and a MOE
        generation schedule. Then, a list 
        of bool values is returned:

        **True**: MoE in layer i

        **False**: FFN in layer i 

        ### The accepted schedules are

        **1)all**: full MoE

        **2)none**: no MoE

        **3) alternate (offset)**: set MoE with step offset

        **4)list[int]**: List of specific indices

        Params:
            layers(int):
                The amount of transformer blocks
            schedule(object):
                The moe embedding behaviour
            offset(int):
                The offset for automatic MoE embedding
                Default value is one

        Returns:
            A tuple of:
                - list[bool]: List representing MoE location
    """
    # Check if casual option matches
    if isinstance(schedule, str):
        cleaned_request = schedule.lower()

        # Check matching
        match cleaned_request:

            case 'all':
                # Return a fully true mask
                return [True]*layers
            case 'none':
                # No MoE included
                return [False]*layers
            case 'alternate':
                return [(i%offset)==0 for i in range(layers)]

    # Case manual embedding is provided -> list of indices
    # 1. Check range valid
    minimum = min(schedule)
    maximum = max(schedule)
    if (minimum>=0) and (maximum<=(layers-1)):
        return [i in schedule for i in range(layers)]
    
    print(f'Mask indices are not within layers range(0-{layers})')