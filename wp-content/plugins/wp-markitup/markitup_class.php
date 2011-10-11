<?php
/*
Copyright (C) 2008  Stefano Verna  (email : stefano.verna@gmail.com)
Based on the great "FCKEditor for Wordpress" plugin by Dean Lee (email : deanlee2@hotmail.com)

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

class TagSet {
	
	function getTagSetName() {
		return 'TagSet';
	}
	function getExternalPluginInfos() {
		return null;
	}
	
	function isExternalPluginActive() {
		return true;
	}
	
	function isExternalPluginInstalled() {
		return true;
	}
	
	function getPluginDir() {
		return dirname(__FILE__) . '/../';
	}
}

$root = dirname(dirname(dirname(dirname(__FILE__))));
if (file_exists($root.'/wp-load.php')) {
	// WP 2.6
	require_once($root.'/wp-load.php');
}

class markitup {
    var $version = '1.0';
    var $default_options = array();
	var $markitup_path = "";
    var $plugin_path ="";

    function trailingslashit($string) {
        if ( '/' != substr($string, -1)) {
            $string .= '/';
        }
        return $string;
    }

	//var $smiley_images = array();
    function markitup()
    {	
		$plugin_dir = basename(dirname(__FILE__));
		$this->siteurl = $this->trailingslashit(get_option('siteurl'));
        load_plugin_textdomain( 'markitup', 'wp-content/plugins/'.$plugin_dir.'/languages');
		
		$this->plugin_path =  $this->siteurl .'wp-content/plugins/' . basename(dirname(__FILE__)) .'/';
		$this->markitup_path = $this->siteurl .'wp-content/plugins/' . basename(dirname(__FILE__)) .'/markitup/';
		$this->markitup_tagsets_path = dirname(__FILE__) . '/markitup/sets/';
		$this->markitup_skins_path = dirname(__FILE__) . '/markitup/skins/';
	
		$this->markitup_tagsets = $this->getTagSets($this->markitup_tagsets_path);
		$this->markitup_skins = $this->getSkins($this->markitup_skins_path);
		
        $this->default_options['editor_height'] = '400';
		$this->default_options['skin'] = 'wp_markitup';
		$this->default_options['tagset'] = 'XHTML';
		$this->default_options['version'] = $version;

		$options = get_option('markitup');
		if (!$options || $options['version'] != $version) {
			if ($options) delete_option("markitup");
			
			add_option('markitup', $this->default_options);
			$options = $this->default_options;
		}
		
		foreach ($options as $option_name => $option_value)
	        $this-> {$option_name} = $option_value;
    }

	function getSkins($dir) {
		$ret = array();
		if ($handle = opendir($dir)) {
		    while (false !== ($file = readdir($handle))) {
		        if ($file == "." || $file == ".." || !is_dir($dir . $file) || !is_file( $dir . $file . '/style.css' )) {
					continue;
				}
				$ret[] = $file;
		    }
		    closedir($handle);
		}
		return $ret;
	}
	
	function getTagSets($dir) {
		$ret = array();
		if ($handle = opendir($dir)) {
		    while (false !== ($file = readdir($handle))) {
		        if ($file == "." || $file == ".." || !is_dir($dir . $file) || !is_file( $dir . $file . '/infos.php' )) {
					continue;
				}
		        include($dir . $file . '/infos.php');
				$ret[] = array(
					'name' => $tagset->getTagSetName(),
					'dir' => $file
				);	
		    }
		    closedir($handle);
		}
		return $ret;
	}

	function deactivate()
	{}

	function activate()
	{
		global $current_user;
		update_user_option($current_user->id, 'rich_editing', 'false', true);
	}

    function option_page()
    {
	 $message = "";
	
		$new_options = array (
			'skin' => (isset($_POST['skin'])?$_POST['skin']:$this->skin),
			'tagset' => (isset($_POST['tagset'])?$_POST['tagset']:$this->tagset),
			'editor_height' => (isset($_POST['height'])?$_POST['height']:$this->editor_height)
		);

		update_option("markitup", $new_options);

		foreach ($new_options as $option_name => $option_value)
	        $this-> {$option_name} = $option_value;
		
		if ($_POST['Submit']) {
			echo '<div class="updated"><p>' . __('Configuration updated!') . '</p></div>';
    	}

	    ?>
		<div class=wrap>
			<h2><?php _e('markItUp! for Wordpress', 'markitup') ?> <?php echo $this->version?></h2>
			<form method="post">
			<table class="form-table">
				<?php if (count($this->markitup_tagsets) > 1): ?>
				<tr valign="top">
					<th scope="row"><?php _e('Tag set to use:', 'markitup'); ?></th>
					<td>
						<fieldset>
							<legend class="hidden"><?php _e('Tag set to use:', 'markitup'); ?></legend>
							<p>
								<label for='tagset'>
									<select name='tagset' id='tagset'>
										<?php
											foreach ($this->markitup_tagsets as $set) {
												echo "<option value='".$set['dir']."'".($set['dir'] == $this->tagset?' selected="selected"':'').">".$set['name']."</option>";
											}
										?>
									</select>
							</p>
						</fieldset>
						<div id="notice">
							
						</div>
						<script>
							(function(){
								var $ = jQuery;
								var check = function() {
									$.get("<?php echo $this->plugin_path?>markitup_class.php", { method: "checkTagSet", name: $("#tagset").val() }, function(data) { $('#notice').html(data);} );
								};
								$("#tagset").change(check);
								check();
							})();
						</script>
					</td>
				</tr>
				<?php endif; ?>
				<?php if (count($this->markitup_skins) > 1): ?>
				<tr valign="top">
					<th scope="row"><?php _e('Skin:', 'markitup'); ?></th>
					<td>
						<fieldset>
							<legend class="hidden"><?php _e('Skin:', 'markitup'); ?></legend>
							<p>
								<label for='skin'>
									<select name='skin' id='skin'>
										<?php
											foreach ($this->markitup_skins as $skin) {
												echo "<option value='$skin'".($skin == $this->skin?' selected="selected"':'').">$skin</option>";
											}
										?>
									</select>
							</p>
						</fieldset>
					</td>
				</tr>
				<?php endif; ?>
				<tr valign="top">
					<th scope="row"><?php _e('Editor Default Height:', 'markitup'); ?></th>
					<td>
						<fieldset>
							<legend class="hidden"><?php _e('Editor Default Height:', 'markitup'); ?></legend>
							<p>
								<label for='height'>
									<input name="height" type="text" id="height" value="<?php echo $this->editor_height?>" size="4" /> px
							</p>
						</fieldset>
					</td>
				</tr>
			</table>
			<p class="submit">
			<input type="submit" name="Submit" value="<?php _e('Save Changes', 'markitup'); ?>" />
			</p>
			</form>
			</div>
		<?php
    }

    function add_admin_head()
    {
    ?>
		<script type="text/javascript" src="<?php echo $this->markitup_path;?>jquery.markitup.pack.js"></script>
		<script type="text/javascript" src="<?php echo $this->markitup_path;?>sets/<?php echo $this->tagset?>/set.php"></script>
		<link rel="stylesheet" type="text/css" href="<?php echo $this->markitup_path;?>skins/<?php echo $this->skin?>/style.css" />
		<link rel="stylesheet" type="text/css" href="<?php echo $this->markitup_path;?>sets/<?php echo $this->tagset?>/style.css" />
	<?php
    }

	function load_markitup()
	{
		?>
	<script type="text/javascript">
		//<![CDATA[
		(function() {
			var $ = jQuery;
			$.get(
				"<?php echo $this->plugin_path?>markitup_class.php", 
				{ 
					method: "checkTagSet", 
					name: "<?php echo $this->tagset?>" 
				}, 
				function(data) { 
					$('#wpbody').prepend($(data));
				}
			);
			$(document).ready(function() {
				if ($("#quicktags").length) {
					$('#content').markItUp(mySettings);
					$("#editorcontainer > div").addClass("<?php echo strtolower($this->tagset)?>");
					$("#quicktags").remove();
				}
			});
		})();
	//]]>
	</script>
		<?php
	}
	
	function getWarnings($name) {
		$info_path = dirname(__FILE__) . '/markitup/sets/' . $name . '/infos.php';
		if (!is_file($info_path)) {
			die();
		}
		include($info_path);
		$needsExt = $tagset->getExternalPluginInfos();
		if (!$needsExt) {
			die();
		}
		if ($tagset->isExternalPluginActive()) {
			die();
		}
		if ($tagset->isExternalPluginInstalled()) {
			$pluginspage = $this->siteurl.'wp-admin/plugins.php';
			$msg = sprintf(__("<strong>Warning:</strong> To correctly use the WP MarkItUp! tagset %s, <strong>you need to activate the <a href='%s' target='_blank'>%s</a></strong> plugin, which seems to be already present in the plugins directory of your Wordpress installation. WP MarkItUp! is just an advanced text editor. It doesn't manage the conversion from the %s syntax to XHTML of your posts. <strong><a href='%s'>Go to the Plugins page &raquo;</a></strong>", 'markitup'), $tagset->getTagSetName(), $needsExt['website'], $needsExt['name'], $tagset->getTagSetName(), $pluginspage);
		} else {
			$msg = sprintf(__("<strong>Warning:</strong> To correctly use the WP MarkItUp! tagset %s, <strong>you need to install and activate the <a href='%s' target='_blank'>%s</a></strong> plugin. WP MarkItUp! is just an advanced text editor. It doesn't manage the conversion from the %s syntax to XHTML of your posts.", 'markitup'), $tagset->getTagSetName(), $needsExt['website'], $needsExt['name'], $tagset->getTagSetName());
		}
		?>
		<div class="error"><p><?php echo $msg?></p></div>
		<?php
	}

    function add_option_page()
    {
		add_options_page('WP MarkItUp!', 'WP MarkItUp!', 'manage_options', 'markitup', array(&$this, 'option_page'));
    }

}

$markitup = new markitup();

	if ($_GET['method'] == 'checkTagSet') {
		echo $markitup->getWarnings($_GET['name']);
	}

?>
