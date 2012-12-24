<?php
		/*
		Plugin Name: Bulk Remove Pending Comments
		Plugin URI: http://allcreatives.net/2011/06/25/wordpress-plugin-bulk-remove-comments/
		Description: Simple plugin to remove any pending comments from your database
		Author: Mike Strand
		Version: 1.0
		Author URI: http://www.allcreatives.net
		*/
		
function brc_admin() { 

	global $wpdb;
	$com_number = $wpdb->query("SELECT comment_ID FROM $wpdb->comments WHERE comment_approved = '0'");
	
	
	if($_POST['brc_hidden'] == 'Y' && $_POST['remove'] == 'Y'){ ?>
    
    <div class="wrap">
			<?php    echo "<h2>Bulk Remove Comments</h2>"; ?>
	
	<?php if($wpdb->query("DELETE FROM $wpdb->comments WHERE comment_approved = '0'") != FALSE){ ?>
	<p style="color:green"><strong>All pending comments have now been removed.</strong></p>
    <?php }else{ ?>
    <p style="color:red"><strong>There was an error, please try again.</strong></p>
    <form name="brc_form" method="post" action="<?php echo str_replace( '%7E', '~', $_SERVER['REQUEST_URI']); ?>">
				<input type="hidden" name="brc_hidden" value="Y">
				
                <input type="checkbox" name="remove" value="Y" /> Remove all pending comments

				<p class="submit">
				<input type="submit" name="Submit" value="Remove" />
				</p>
			</form>

<?php
}
echo "</div>";
	}else if($_POST['brc_hidden'] == 'Y' && $_POST['remove'] != 'Y'){ ?>
	
    
	<div class="wrap">
			<?php    echo "<h2>Bulk Remove Comments</h2>"; ?>
			<p><strong>Note: Once you have ticked the box below and click remove all pending comments will be removed from your database.</strong></p>
            <p style="color:red"><strong>Please tick the checkbox if you would like to remove pending comments.</strong></p>
			<form name="brc_form" method="post" action="<?php echo str_replace( '%7E', '~', $_SERVER['REQUEST_URI']); ?>">
				<input type="hidden" name="brc_hidden" value="Y">
				
                <input type="checkbox" name="remove" value="Y" /> Remove all pending comments

				<p class="submit">
				<input type="submit" name="Submit" value="Remove" />
				</p>
			</form>
		
    
    <?php
	
	}else{

?>
			
			
            
            		<div class="wrap">
			<?php    echo "<h2>Bulk Remove Comments (" . $com_number . " Pending)</h2>" ; ?>
			<?php if($com_number > 0){ ?>
            <p><strong>Note: Once you have ticked the box below and click remove all pending comments will be removed from your database.</strong></p>
            
            
            
			<form name="brc_form" method="post" action="<?php echo str_replace( '%7E', '~', $_SERVER['REQUEST_URI']); ?>">
				<input type="hidden" name="brc_hidden" value="Y">
				
                <input type="checkbox" name="remove" value="Y" /> Remove all pending comments

				<p class="submit">
				<input type="submit" name="Submit" value="Remove" />
				</p>
			</form>
		</div>
        
        <?php }else{ ?>
        
        <p><strong>You have 0 pending comments to remove.</strong></p>
        
        <?php } ?>
	
			
<?php
}

}

function brc_admin_actions() {
			add_management_page("Bulk Remove Comments", "Bulk Remove Comments", 1, "Bulk_Remove_Comments", "brc_admin");
}

		add_action('admin_menu', 'brc_admin_actions');
?>