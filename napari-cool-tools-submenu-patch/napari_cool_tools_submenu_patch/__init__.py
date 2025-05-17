from napari._app_model import get_app_model
from napari._app_model.constants import MenuGroup, MenuId
from app_model.types import SubmenuItem, ToggleRule

app = get_app_model()
menu = list(app.menus._menu_items['napari/plugins'].keys())

#Parameters to set the patch
cool_tools_menuid = 'napari/plugins/cool-tools'
tool_title = 'COOL LAB Tools'

#this is the unique plugin id for coolt tools
plugin_id = 'cool-tools'

for item in menu:
    
    #if a contributable
    if item.group is MenuGroup.PLUGIN_SINGLE_CONTRIBUTIONS:
        
        #TODO: Move the Items to the new submenu
        #only move the desired plugin
        if plugin_id not in item.command.id:
            continue
        
        app.menus._menu_items['napari/plugins'].pop(item)
        new_item = [
            (
                cool_tools_menuid,
                item,
            ),
        ]
                
        app.menus.append_menu_items(new_item)
        
    #if multiple contributable
    if item.group is MenuGroup.PLUGIN_MULTI_SUBMENU:
        
        if plugin_id not in item.submenu:
            continue
                
        app.menus._menu_items['napari/plugins'].pop(item)
        
        if item.submenu in app.menus._menu_items:
            submenu = list(app.menus._menu_items[item.submenu].keys())
            app.menus._menu_items.pop(item.submenu)
            
            #TODO: Move the Items to the new submenu
            for subitem in submenu:
                new_subitem = [
                    (
                        f'{cool_tools_menuid}/{item.title}',
                        subitem,
                    ),
                ]
                        
                app.menus.append_menu_items(new_subitem)
                
            #TODO: Move the Submenu
            new_submenu = [
                (
                    cool_tools_menuid,
                    SubmenuItem(
                        submenu= f'{cool_tools_menuid}/{item.title}',
                        title=item.title,
                        group=item.group,
                    ),
                ),
            ]
                    
            app.menus.append_menu_items(new_submenu)
            
        
        # item.submenu = 
    

submenu = [
    (
        MenuId.MENUBAR_PLUGINS,
        SubmenuItem(
            submenu=cool_tools_menuid,
            title=tool_title,
            group=MenuGroup.PLUGIN_MULTI_SUBMENU,
        ),
    ),
]

app.menus.append_menu_items(submenu)
      
#update it on the GUI
app.menus.menus_changed.emit('napari/plugins')